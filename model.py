import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from utils import generate_engineered_features

class DualEncoder(nn.Module):
    '''
    A PyTorch binary classifier for classifying a pair of sentences as being from the same author or not.

    The model creates two 768 length embeddings for each sentence using an identical encoder
    architecture (by default DeBERTa base) and uses these embeddings to calculate distance and uses
    a simple binary threshold to classify. The threshold is 0.5 by default.
    '''
    def __init__(self, model_name="microsoft/deberta-base", device="cuda", distance_metric="cosine", margin=1.0,
                attention_dropout=0.1, hidden_dropout=0.1):
        '''
        Initializes the model.

        :model_name (str): Path to the model, either for Hugging Face or local directory
        :device (str): the name of the device for training and classification as a string
        :distance_metric (str): the metric for distance to use for contrastive loss (options are cosine and euclidean)
        :margin (float): margin hyperparameter for contrastive loss
        '''
        super(DualEncoder, self).__init__()
        
        # Load the pre-trained language model
        config = AutoConfig.from_pretrained(model_name)
        config.attention_probs_dropout_prob = attention_dropout  # Dropout for attention probabilities
        config.hidden_dropout_prob = hidden_dropout  # Dropout for hidden layers
        self.deberta = AutoModel.from_pretrained(model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.to(device)
        self.device = device
        
        # Hyperparameters for contrastive loss
        self.margin = margin

        # Initial threshold for classification
        if distance_metric == "cosine":
            self.threshold = 0.0
        else:
            self.threshold = 0.5

        # Set distance metric (note: threshold direction flips for euclidean distance)
        if distance_metric == "cosine":
            self.distance_metric = "cosine"
        elif distance_metric == "euclidean":
            self.distance_metric = "euclidean"
        else:
            print(f"Invalid distance metric {distance_metric}, setting to default of cosine.")
            self.distance_metric = "cosine"
        
    def forward_once(self, input_ids, attention_mask):
        # Pass input through DeBERTa to get the pooled output embedding
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use the CLS token representation
        return pooled_output

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # Generate embeddings for both text spans
        embedding1 = self.forward_once(input_ids1, attention_mask1)
        embedding2 = self.forward_once(input_ids2, attention_mask2)
        
        # Return the embeddings for contrastive loss
        return embedding1, embedding2
    
    def tokenize(self, texts):
        '''
        Tokenizes a given sequence of strings in the format "sentence1 [SNIPPET] sentence2"

        :param texts ([str]): a sequence of strings which are pairs of sentences separated by the special token [SNIPPET]

        :return Tuple(Tensor): 4 tensors, with the first 2 being the input ids (from vocab.txt) and attention mask
            for the first sentence, and the last 2 being the input ids and attention mark for the second sentence
        '''
        first_sentences = []
        second_sentences = []
        for text_pair in texts:
            sentences = text_pair.split(" [SNIPPET] ")
            first_sentences.append(sentences[0])
            second_sentences.append(sentences[1])

        inputs1 = self.tokenizer(first_sentences, padding=True, truncation=True, return_tensors="pt")
        inputs2 = self.tokenizer(second_sentences, padding=True, truncation=True, return_tensors="pt")

        return inputs1["input_ids"].to(self.device), inputs1["attention_mask"].to(self.device), \
               inputs2["input_ids"].to(self.device), inputs2["attention_mask"].to(self.device)

    def compute_contrastive_loss(self, embedding1, embedding2, labels):

        if self.distance_metric == "cosine":
            distances = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        elif self.distance_metric == "euclidean":
            distances = torch.nn.functional.pairwise_distance(embedding1, embedding2)
        
        # Contrastive Loss: Pull similar pairs close and push dissimilar pairs apart
        if self.distance_metric == "cosine":
            loss_contrastive = torch.mean(
                (1 - labels) * torch.pow(distances + 1, 2) + # Different authors
                labels * torch.pow(distances - 1, 2) # Same authors
            )
        elif self.distance_metric == "euclidean":
            loss_contrastive = torch.mean(
                (1 - labels) * torch.pow(distances, 2) +
                labels * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
            )
        return loss_contrastive
    
    def predict(self, texts, return_distance=False):
        predictions = []
        distances = []
        for text_pair in texts:
            input_ids1, attention_mask1, \
            input_ids2, attention_mask2 = self.tokenize([text_pair])

            with torch.no_grad():
                embedding1, embedding2 = self(input_ids1, attention_mask1, input_ids2, attention_mask2)

                if self.distance_metric == "cosine":
                    distance = torch.nn.functional.cosine_similarity(embedding1, embedding2)
                elif self.distance_metric == "euclidean":
                    distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)

            distances.append(distance)

            if self.distance_metric == "cosine":
                if distance > self.threshold:
                    predictions.append(1)
                else:
                    predictions.append(0)
            elif self.distance_metric == "euclidean":
                if distance < self.threshold:
                    predictions.append(1)
                else:
                    predictions.append(0)

        if return_distance:
            return predictions, distances
        else:
            return predictions

class AuthorClassifier(nn.Module):
    def __init__(self, dual_encoder, model_name="microsoft/deberta-base", device="cuda",
                 attention_dropout=0.1, hidden_dropout=0.1, embedding_size=768, freeze_dual_encoder=True,
                use_engineered_features=False, num_engineered_features=0, common_tokens=None):
        super(AuthorClassifier, self).__init__()

        self.dual_encoder = dual_encoder

        # Load the pre-trained language model
        config = AutoConfig.from_pretrained(model_name)
        config.attention_probs_dropout_prob = attention_dropout  # Dropout for attention probabilities
        config.hidden_dropout_prob = hidden_dropout  # Dropout for hidden layers
        self.deberta = AutoModel.from_pretrained(model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.to(device)
        self.device = device

        self.freeze_dual_encoder = freeze_dual_encoder
        self.use_engineered_features = use_engineered_features
        self.common_tokens = common_tokens

        # Final linear classification layer with 2 embeddings from dual encoder fine-tuned for distance,
        # 2 embeddings from fine-tuned DeBERTa for binary classification, and engineered features
        self.classifier = nn.Linear(embedding_size*4+num_engineered_features, 1)

    def forward_once(self, input_ids, attention_mask):
        # Pass input through DeBERTa to get the pooled output embedding
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use the CLS token representation
        return pooled_output

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, engineered_features=None):
        # Generate distance-optimized embeddings for both text spans
        if self.freeze_dual_encoder:
            with torch.no_grad():
                distance_embedding1, distance_embedding2 = self.dual_encoder(input_ids1, attention_mask1, input_ids2, attention_mask2)
        else: 
            distance_embedding1, distance_embedding2 = self.dual_encoder(input_ids1, attention_mask1, input_ids2, attention_mask2)
        
        # Generate trained embeddings
        embedding1 = self.forward_once(input_ids1, attention_mask1)
        embedding2 = self.forward_once(input_ids2, attention_mask2)

        distance_embedding1 = distance_embedding1.to(self.device)
        distance_embedding2 = distance_embedding2.to(self.device)
        embedding1 = embedding1.to(self.device)
        embedding2 = embedding2.to(self.device)

        self.classifier.to(self.device)
        
        # Return the labels
        if self.use_engineered_features:
            concat = torch.cat((distance_embedding1, distance_embedding2, embedding1, embedding2, engineered_features), 1).to(self.device)
        else:
            concat = torch.cat((distance_embedding1, distance_embedding2, embedding1, embedding2), 1).to(self.device)
        # return torch.argmax(torch.nn.functional.softmax(self.classifier(concat), dim=1), dim=1)
        return self.classifier(concat).to(self.device)
    
    def tokenize(self, texts):
        '''
        Tokenizes a given sequence of strings in the format "sentence1 [SNIPPET] sentence2"

        :param texts ([str]): a sequence of strings which are pairs of sentences separated by the special token [SNIPPET]

        :return Tuple(Tensor): 4 tensors, with the first 2 being the input ids (from vocab.txt) and attention mask
            for the first sentence, and the last 2 being the input ids and attention mark for the second sentence
        '''
        first_sentences = []
        second_sentences = []
        for text_pair in texts:
            sentences = text_pair.split(" [SNIPPET] ")
            first_sentences.append(sentences[0])
            second_sentences.append(sentences[1])

        inputs1 = self.tokenizer(first_sentences, padding=True, truncation=True, return_tensors="pt")
        inputs2 = self.tokenizer(second_sentences, padding=True, truncation=True, return_tensors="pt")

        return inputs1["input_ids"].to(self.device), inputs1["attention_mask"].to(self.device), \
               inputs2["input_ids"].to(self.device), inputs2["attention_mask"].to(self.device)
    
    def predict(self, texts_or_features, use_features=False, nlp=None, batch_size=8, device=None):
        if device:
            self.to(device)
            self.dual_encoder.to(device)
            self.device = device
            self.dual_encoder.device = device
            
        with torch.no_grad():
            if use_features:
                if self.use_engineered_features:
                    logits = self(*texts_or_features[:4], engineered_features=texts_or_features[4])
                else:
                    logits = self(*texts_or_features)
            else:
                logits = None
                for i in range(0, len(texts_or_features), batch_size):
                    input_ids1, attention_masks1, input_ids2, attention_masks2 = self.tokenize(texts_or_features[i:min(i+batch_size, len(texts_or_features))])
                    if self.use_engineered_features:
                        cur_logits = self(input_ids1, attention_masks1,
                                          input_ids2, attention_masks2,
                                          engineered_features=generate_engineered_features(texts_or_features[i:min(i+batch_size, len(texts_or_features))], nlp, self.common_tokens))
                    else:
                        cur_logits = self(input_ids1, attention_masks1, input_ids2, attention_masks2)
                    
                    if logits == None:
                        logits = cur_logits
                    else:
                        logits = torch.cat((logits, cur_logits))
        
        probabilities = torch.sigmoid(logits)
        return (probabilities > 0.5).long().squeeze()