import argparse
import re
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import DualEncoder, AuthorClassifier
from utils import SentencesDataset, eval_model



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read in train and dev set
    df = pd.read_csv(args.data_path)

    train_df, dev_df = train_test_split(df, test_size=0.25, random_state=42) # Split into 75/25 train/dev

    dev_df, test_df = train_test_split(dev_df, test_size=0.6, random_state=42) # Further split for 75/15/10 train/dev/test

    torch.manual_seed(42)

    # Initialize model, optimizer, and loss function
    model = DualEncoder(model_name = args.dual_encoder_model_path, device=device, distance_metric=args.distance_metric)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_dataset = SentencesDataset(train_df["LABEL"], train_df["TEXT"], model.tokenize, device=device)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print("Training Dual Encoder")
    for epoch in range(args.num_epochs_dual_encoder):
        model.train()

        for features, labels in dataloader:
            batch_ids1 = features[0]
            batch_masks1 = features[1]
            batch_ids2 = features[2]
            batch_masks2 = features[3]
            
            # Forward pass for a batch of pairs
            embeddings1, embeddings2 = model(batch_ids1, batch_masks1, batch_ids2, batch_masks2)

            # Calculate contrastive loss
            loss = model.compute_contrastive_loss(embeddings1, embeddings2, labels)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        if args.report_per_epoch:
            model.eval()

            predictions, distances = model.predict(dev_df["TEXT"], return_distance=True)
            
            print(f"Accuracy: {accuracy_score(dev_df['LABEL'], predictions)}\nPrecision: {precision_score(dev_df['LABEL'], predictions)}\nRecall: {recall_score(dev_df['LABEL'], predictions)}\nF1: {f1_score(dev_df['LABEL'], predictions)}")
            stats = [[0,0],[0,0]]
            for i in range(len(predictions)):
                pred = predictions[i]
                true = dev_df["LABEL"].iloc[i]
                stats[pred][true] += 1
            print(f"TP: {stats[1][1]},FP: {stats[1][0]},TN: {stats[0][0]},FN: {stats[0][1]}")

            not_same = []
            same = []
            for i, label in enumerate(dev_df["LABEL"]):
                if label:
                    same.append(distances[i].cpu().item())
                else:
                    not_same.append(distances[i].cpu().item())
            
            print(f"Label 0 Mean: {np.mean(not_same)}\nLabel 1 Mean: {np.mean(same)}\n==============================")
    
    print("Dual Encoder Report:")
    model.eval()

    predictions, distances = model.predict(dev_df["TEXT"], return_distance=True)
    
    print(f"Accuracy: {accuracy_score(dev_df['LABEL'], predictions)}\nPrecision: {precision_score(dev_df['LABEL'], predictions)}\nRecall: {recall_score(dev_df['LABEL'], predictions)}\nF1: {f1_score(dev_df['LABEL'], predictions)}")
    stats = [[0,0],[0,0]]
    for i in range(len(predictions)):
        pred = predictions[i]
        true = dev_df["LABEL"].iloc[i]
        stats[pred][true] += 1
    print(f"TP: {stats[1][1]},FP: {stats[1][0]},TN: {stats[0][0]},FN: {stats[0][1]}")

    not_same = []
    same = []
    for i, label in enumerate(dev_df["LABEL"]):
        if label:
            same.append(distances[i].cpu().item())
        else:
            not_same.append(distances[i].cpu().item())
    
    print(f"Label 0 Mean: {np.mean(not_same)}\nLabel 1 Mean: {np.mean(same)}\n==============================")
    # Previous model is now a frozen part of the full model
    dual_encoder = model
        
    torch.save(dual_encoder.state_dict(), "dual_encoder.pt")
    print("Dual encoder saved.")

    # Second Stage of Training

    if device == "cuda":
        spacy.require_gpu()

    if args.use_engineered_features:
        nlp = spacy.load("en_core_web_trf")

        common_words_file_path = "en_5k.txt"
        with open(common_words_file_path, 'r') as f:
            common_tokens = set(f.read().split("\n"))

    if args.use_engineered_features:
        model = AuthorClassifier(dual_encoder, model_name = args.main_model_path, device=device, embedding_size=args.embedding_size,
                            attention_dropout=args.attention_dropout, hidden_dropout=args.hidden_dropout, freeze_dual_encoder=args.freeze_dual_encoder,
                            use_engineered_features=True, num_engineered_features=6, common_tokens=common_tokens)
    else:
        model = AuthorClassifier(dual_encoder, model_name = args.main_model_path, device=device, embedding_size=args.embedding_size,
                            attention_dropout=args.attention_dropout, hidden_dropout=args.hidden_dropout, freeze_dual_encoder=args.freeze_dual_encoder)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=args.weight_decay)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.positive_weight]).to(device))

    if args.load_from_dataset:
        with open(Path(args.dataset_path) / "train_dataset.pickle", "rb") as f:
            train_dataset = pickle.load(f)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        with open(Path(args.dataset_path) / "dev_dataset.pickle", "rb") as f:
            dev_dataset = pickle.load(f)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.use_engineered_features:
        train_dataset = SentencesDataset(train_df["LABEL"], train_df["TEXT"], model.tokenize, nlp=nlp, common_tokens=common_tokens, device=device)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        dev_dataset = SentencesDataset(dev_df["LABEL"], dev_df["TEXT"], model.tokenize, nlp=nlp, common_tokens=common_tokens, device=device)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        train_dataset = SentencesDataset(train_df["LABEL"], train_df["TEXT"], model.tokenize, device=device)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        dev_dataset = SentencesDataset(dev_df["LABEL"], dev_df["TEXT"], model.tokenize, device=device)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    if args.save_dataset:
        with open(Path(args.dataset_path) / 'train_dataset.pickle', 'wb') as f:
            pickle.dump(train_dataset, f)

        with open(Path(args.dataset_path) / 'dev_dataset.pickle', 'wb') as f:
            pickle.dump(dev_dataset, f)

    print("Training Full Model")
    for epoch in range(args.num_epochs):
        model.train()

        for features, labels in dataloader:
            batch_ids1 = features[0]
            batch_masks1 = features[1]
            batch_ids2 = features[2]
            batch_masks2 = features[3]
            
            # Forward pass for a batch of pairs
            if args.use_engineered_features:
                pred_logits = model(batch_ids1, batch_masks1, batch_ids2, batch_masks2, engineered_features=features[4])
            else:
                pred_logits = model(batch_ids1, batch_masks1, batch_ids2, batch_masks2)

            # Calculate loss
            loss = loss_function(pred_logits.squeeze(), labels.float())

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        if args.report_per_epoch:
            if args.use_engineered_features:
                eval_model(model, dev_dataloader, use_dataloader=True, nlp=nlp, device=device)
            else:
                eval_model(model, dev_dataloader, use_dataloader=True, device=device)
    
    torch.save(model.state_dict(), "model.pt")
    print("Model saved.")
    if args.use_engineered_features:
        eval_model(model, dev_dataloader, use_dataloader=True, nlp=nlp, device=device)
    else:
        eval_model(model, dev_dataloader, use_dataloader=True, device=device)
    
    if args.eval_test:
        if args.use_engineered_features:
            eval_model(model, test_df, nlp=nlp)
        else:
            eval_model(model, test_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/train.csv", help="file path to complete dataset")
    parser.add_argument("--dual_encoder_model_path", default="distilbert/distilbert-base-cased", help="The path to the model to use for the dual encoder")
    parser.add_argument("--main_model_path", default="distilbert/distilbert-base-cased", help="The path to the model to use for the main model")
    parser.add_argument("--common_tokens_path", default="en_5k.txt", help="A path to a list of the 5,000 most common words in English")
    parser.add_argument("--distance_metric", choices=["cosine", "euclidean"], default="cosine", help="The distance metric to use for training distance embeddings (cosine|euclidean)")
    parser.add_argument("--embedding_size", type=int, default="768", help="Size of the sentence embeddings for your model")

    parser.add_argument("--batch_size_dual_encoder", type=int, default=8, help="The batch size for training the dual encoder")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size for training")
    parser.add_argument("--num_epochs_dual_encoder", type=int, default=10, help="The number of epochs for training the dual encoder layer")
    parser.add_argument("--num_epochs", type=int, default=10, help="The number of epochs for training the full model (second stage of training)")

    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout parameter for attention layers")
    parser.add_argument("--hidden_dropout", type=float, default=0.1, help="Dropout parameter for hidden layers")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay parameter for full model")
    parser.add_argument("--positive_weight", type=float, default=1.0, help="Weight for loss function for the positive class")

    parser.add_argument("--freeze_dual_encoder", help="Do not adjust weights of dual encoder is full model training", action="store_true")
    parser.add_argument("--use_engineered_features", help="Use additional engineered features", action="store_true")
    parser.add_argument("--save_dataset", help="Save datasets w/ features once generated", action="store_true")
    parser.add_argument("--load_from_dataset", help="Load features from a pickled dataset", action="store_true")
    parser.add_argument("--dataset_path", default=".", help="Directory to save and load datasets")

    parser.add_argument("--report_per_epoch", help="Eval every epoch if present", action="store_true")
    parser.add_argument("--eval_test", help="Eval using test data at end if present", action="store_true")

    args = parser.parse_args()

    main(args)