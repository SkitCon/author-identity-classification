import argparse
import re
import pandas as pd
import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import DualEncoder, AuthorClassifier
from train import eval_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data_path)

    if args.use_engineered_features:
        nlp = spacy.load("en_core_web_trf")

        common_words_file_path = "en_5k.txt"
        with open(common_words_file_path, 'r') as f:
            common_tokens = set(f.read().split("\n"))
    else:
        nlp = None

    dual_encoder = DualEncoder(model_name = args.dual_encoder_model_path, device=device, distance_metric=args.distance_metric)

    if args.use_engineered_features:
        model = AuthorClassifier(dual_encoder, model_name = args.main_model_path, device=device,
                                 embedding_size=args.embedding_size, use_engineered_features=True, num_engineered_features=6, common_tokens=common_tokens)
    else:
        model = AuthorClassifier(dual_encoder, model_name = args.main_model_path, device=device, embedding_size=args.embedding_size)
    model.load_state_dict(torch.load(args.model_path, weights_only=False))

    model.eval()

    if args.eval:
        eval_model(model, df, nlp=nlp)
    
    predictions = model.predict(df["TEXT"], nlp=nlp)

    out_df = pd.DataFrame({"ID":df["ID"], "LABEL":predictions.cpu()})
    out_df.to_csv("predicted.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="model.pt", help="Path to the saved model")
    parser.add_argument("--data_path",  default="data/test.csv", help="Path to the data to run inference on")
    parser.add_argument("-common_tokens_path", default="en_5k.txt", help="A path to a list of the 5,000 most common words in English")

    parser.add_argument("--dual_encoder_model_path", default="distilbert/distilbert-base-cased", help="The path to the model to use for the dual encoder")
    parser.add_argument("--main_model_path", default="distilbert/distilbert-base-cased", help="The path to the model to use for the main model")
    parser.add_argument("--distance_metric", choices=["cosine", "euclidean"], default="cosine", help="The distance metric to use for training distance embeddings (cosine|euclidean)")
    parser.add_argument("--embedding_size", type=int, default="768", help="Size of the sentence embeddings for your model")
    
    parser.add_argument("--use_engineered_features", help="Use additional engineered features", action="store_true")
    parser.add_argument("--eval", help="If present, try to evaluate based on \"LABEL\" column", action="store_true")

    args = parser.parse_args()

    main(args)
