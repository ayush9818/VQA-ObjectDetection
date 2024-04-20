from utils import load_clip
import argparse 
import os
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-image-dir", type=str)
    parser.add_argument("--val-image-dir", type=str)
    parser.add_argument("--train-file-path", type=str)
    parser.add_argument("--val-file-path", type=str)
    parser.add_argument("--clip-model", type=str, default="RN50")
    parser.add_argument("--save-dir", type=str)

    args = parser.parse_args()
    return args


def extract_embeddings(image_dir, df_path, device):
    df = pd.read_csv(df_path)
    image_path = df

if __name__ == "__main__":
    pass