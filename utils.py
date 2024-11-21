import re
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SentencesDataset(Dataset):
    def __init__(self, labels, texts, tokenizer, nlp=None, common_tokens=None, device="cuda"):
        self.labels = torch.tensor(list(labels), dtype=torch.long).to(device)
        self.input_ids1, self.attention_masks1, self.input_ids2, self.attention_masks2 = tokenizer(texts)
        if nlp:
            self.use_engineered_features = True
            self.engineered_features = generate_engineered_features(texts, nlp, common_tokens).to(device)
        else:
            self.use_engineered_features = False
            self.engineered_features = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.use_engineered_features:
            return [self.input_ids1[i], self.attention_masks1[i], self.input_ids2[i], self.attention_masks2[i], self.engineered_features[i]], self.labels[i]
        else:
            return [self.input_ids1[i], self.attention_masks1[i], self.input_ids2[i], self.attention_masks2[i]], self.labels[i]
    
def count_pos(doc, parts_of_speech=[r"NOUN.*", r"VERB.*"]):
  '''
  Count parts of speech in the sentence represented by doc.

  :param: doc (SpaCy Doc): the document to be analyzed
  :param: parts_of_speech ([Str]): an array of regex strings to define parts of speech to be counted, by default counts nouns and verbs

  :return: an array of integers representing the counts of each part of speech in the sentence, respective to the order of the regex array
  '''
  if doc.text == "":
    return [None] * len(parts_of_speech)
  counts = [0] * len(parts_of_speech)
  for token in doc:
    for i, part_of_speech in enumerate(parts_of_speech):
      if re.match(part_of_speech, token.pos_):
        counts[i] += 1
  return counts

def measure_rarity(doc, common_tokens):
  '''
  Measure the number of tokens in the text which are rare in both a general corpus and political (genre) corpus.

  :param: doc (SpaCy Doc): the document to be analyzed

  :return: int, int: the number of rare tokens in the sentence as compared to a general corpus and the number of rare tokens in the sentence as compared to a political corpus
  '''
  rare_token_count = 0
  punct_count = 0
  for token in doc:
    text = re.sub(r"[^\w\s]", "", token.text)
    if text != "":
      if not token.text.lower() in common_tokens:
        rare_token_count += 1
    else:
      punct_count += 1
  return rare_token_count / (len(doc) - punct_count)
    
def generate_engineered_features(texts, nlp, common_tokens, device="cuda"):
    
    first_sentences, second_sentences = zip(*texts.apply(lambda x: x.split(" [SNIPPET] ")))
    
    print(f"Generating {len(texts)} spacy docs")
    
    spacy1 = [nlp(sentence) for sentence in first_sentences]
    spacy2 = [nlp(sentence) for sentence in second_sentences]

    noun_prop1 = torch.tensor([count_pos(x, parts_of_speech=[r"NOUN.*"])[0] / len(x) for x in spacy1], dtype=torch.float16).to(device)
    noun_prop2 = torch.tensor([count_pos(x, parts_of_speech=[r"NOUN.*"])[0] / len(x) for x in spacy2], dtype=torch.float16).to(device)
    noun_prop_diff = ((noun_prop1 - noun_prop2)**2)**0.5

    rarity1 = torch.tensor([measure_rarity(x, common_tokens=common_tokens) / len(x) for x in spacy1], dtype=torch.float16).to(device)
    rarity2 = torch.tensor([measure_rarity(x, common_tokens=common_tokens) / len(x) for x in spacy2], dtype=torch.float16).to(device)
    rarity_diff = ((rarity1 - rarity2)**2)**0.5

    return torch.cat((noun_prop1.reshape(-1,1), noun_prop2.reshape(-1,1), noun_prop_diff.reshape(-1,1), rarity1.reshape(-1,1), rarity2.reshape(-1,1), rarity_diff.reshape(-1,1)), 1)

def eval_model(model, df_or_dataloader, nlp=None, use_dataloader=False, device="cuda"):
    model.eval()

    if use_dataloader:
        predictions = torch.tensor([], dtype=int).to(device)
        labels = []
        for features, cur_labels in df_or_dataloader:
            cur_predictions = model.predict(features, use_features=True, nlp=nlp, device=device)
            predictions = torch.cat((predictions, cur_predictions))
            labels += list(cur_labels.cpu())
    else:
        predictions = model.predict(df_or_dataloader["TEXT"], nlp=nlp)
        labels = list(df_or_dataloader["LABEL"])

    predictions = predictions.cpu()
        
    print(f"Accuracy: {accuracy_score(labels, predictions)}\nPrecision: {precision_score(labels, predictions)}\nRecall: {recall_score(labels, predictions)}\nF1: {f1_score(labels, predictions)}")
    stats = [[0,0],[0,0]]
    for i in range(len(predictions)):
        pred = predictions[i]
        true = labels[i]
        stats[pred][true] += 1
    print(f"TP: {stats[1][1]},FP: {stats[1][0]},TN: {stats[0][0]},FN: {stats[0][1]}\n===================================")