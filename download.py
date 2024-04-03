# This file runs during container build time to get model weights built into the container
import torch
from transformers import AutoTokenizer

from utils.custom_model import CustomModel


def download_model():
    # do a dry run of loading the model, which will download weights
    torch.load('model_7', "cpu")
    AutoTokenizer.from_pretrained("vinai/bertweet-large")

if __name__ == "__main__":
    download_model()
