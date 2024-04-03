import torch
import runpod

from transformers import AutoTokenizer
from utils.TweetNormalizer import normalizeTweet
from utils.custom_model import CustomModel

def handler(event):
    model = torch.load('model_7')
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")

    context = {
        "model": model,
        "tokenizer": tokenizer
    }

    return process(event['input'], context)

def process(request, context):
    prompt = request["text"]
    model = context.get("model")
    tokenizer = context.get('tokenizer')
    
    text = [normalizeTweet(prompt)]
    text = tokenizer(text, padding='longest')
    text = {k: torch.tensor(v, device='cuda:0') for k, v in text.items()}
    
    out = model.forward(**text)
    for i, t in zip(out, prompt):
        res = {'text': prompt, 'hate': i[0].tolist(), 'abusing': i[1].tolist(), 'neutral': i[2].tolist(), 'spam': i[3].tolist()}

    return {"outputs": res}

if __name__ == "__main__":
    runpod.serverless.start({'handler': handler})
