from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
#import pandas as pd
import numpy as np
from fastapi import Request,FastAPI
from pydantic import BaseModel
import uvicorn

def semanticScore(tokenizer, model, sentences):
    # initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentences:
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt').to(torch_device)
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    
    # getting last_hidden_state using mean pooling to calculate cosine similarity
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    mean_pooled = torch.sum(masked_embeddings, 1) / torch.clamp(mask.sum(1), min=1e-9)
    # convert from PyTorch tensor to numpy array
    mean_pooled = mean_pooled.detach().numpy()

    return mean_pooled

def textClean(text):
    import string
    return text.translate(text.maketrans('', '', string.punctuation)).lower()

#sentences = ['A Type I error is a false positive (claiming something has happened when it hasn’t), and a Type II error is a false negative (claiming nothing has happened when it actually has).',
#            'A type I error (false-positive) occurs if an investigator rejects a null hypothesis that is actually true in the population; a type II error (false-negative) occurs if the investigator fails to reject a null hypothesis that is actually false in the population.',
#            'In statistical hypothesis testing, a type I error is the mistaken rejection of an actually true null hypothesis (also known as a "false positive" finding or conclusion; example: "an innocent person is convicted"), while a type II error is the mistaken acceptance of an actually false null hypothesis (also known as a "false negative" finding or conclusion; example: "a guilty person is not convicted").',
#            'A Type I error is a false positive, and a Type II error is a false negative.']
# answer is at index 0

app = FastAPI()

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)

class SummaryRequest(BaseModel):
    text: str

model = AutoModel.from_pretrained('./bert_model')
tokenizer = AutoTokenizer.from_pretrained('./bert_model/')
model = model.to(torch_device)
answer = 'A Type I error is a false positive (claiming something has happened when it hasn’t), and a Type II error is a false negative (claiming nothing has happened when it actually has).'
sentences = [answer]

@app.get('/')
async def home():
    return {"message": "Hello World"}
    
@app.post("/summary")
async def getsummary(user_request_in: SummaryRequest):
    sentences.append(user_request_in.text)
    mean_pooled = semanticScore(tokenizer, model, sentences)
    semantic_scores = [round(score*100, 2) for score in cosine_similarity([mean_pooled[0]], mean_pooled).reshape(-1,)]
#print(semantic_scores)
    return semantic_scores[1]
