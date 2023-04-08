from fastapi import FastAPI,Request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from fastapi.middleware.cors import CORSMiddleware
from divideText import get_chunks
from pymongo import MongoClient
from cosine import cosineSimilarity
from retriveData import fetchData
from relevantDocument import findRelevantDocument
import torch
app=FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

text=""
databaseName = 'chatbot'

# connect with authentication
client = MongoClient("mongodb+srv://sidharthbansal2301:bDrTm2aK2OXUi8Mq@chatbot.kafcl1k.mongodb.net/?retryWrites=true&w=majority")
dbCapstone=client.Capstone
collection=dbCapstone.Documents


def bert(question,data):

  input_ids = tokenizer.encode(question, data)

  tokens = tokenizer.convert_ids_to_tokens(input_ids)

  sep_idx = input_ids.index(tokenizer.sep_token_id)

  num_seg_a = sep_idx+1

  num_seg_b = len(input_ids) - num_seg_a

  segment_ids = [0]*num_seg_a + [1]*num_seg_b

  assert len(segment_ids) == len(input_ids)

  output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
  start_scores = output.start_logits.detach().numpy().flatten()
  end_scores = output.end_logits.detach().numpy().flatten()
  startMax=max(start_scores)
  endMax=max(end_scores)

  answer_start = torch.argmax(output.start_logits)
  answer_end = torch.argmax(output.end_logits)


  if answer_end >= answer_start:
    answer = tokens[answer_start]
    for i in range(answer_start+1, answer_end+1):
      if tokens[i][0:2] == "##":
        answer += tokens[i][2:]
      else:
        answer += " " + tokens[i]
  else:
    return {"answer":"Unable to find the answer to your question.","weight":-1}

  if answer.startswith("[CLS]"):
    answer = "Unable to find the answer to your question."
  return {"answer":answer,"weight":endMax+startMax}

@app.post("/question")
async def root(qs:Request):
  q=await qs.json()
  question=q["question"]
  documentsDic=fetchData(collection)
  maxVal=-1
  ans=""
  relevantDocuments=findRelevantDocument(documentsDic,question)
  print(relevantDocuments)
  for doc in relevantDocuments:
    pair=bert(question,doc.document)
    print(pair)
    w=pair["weight"]
    if w>maxVal:
      maxVal=w
      ans=pair["answer"]
  print(ans)
  return ans



@app.post('/upload')
async def uploadData(data:Request):
  obj=await data.json()
  text=obj["data"]
  chuncks=get_chunks(text,400)
  for document in chuncks:
    collection.insert_one({"document":document})










