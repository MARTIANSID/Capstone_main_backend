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


@app.post("/question")
async def root(qs:Request):
  q=await qs.json()
  question=q["question"]
  input_ids = tokenizer.encode(question, text)

  tokens = tokenizer.convert_ids_to_tokens(input_ids)

  sep_idx = input_ids.index(tokenizer.sep_token_id)

  num_seg_a = sep_idx+1

  num_seg_b = len(input_ids) - num_seg_a

  segment_ids = [0]*num_seg_a + [1]*num_seg_b

  assert len(segment_ids) == len(input_ids)

  output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

  answer_start = torch.argmax(output.start_logits)
  answer_end = torch.argmax(output.end_logits)


  if answer_end >= answer_start:
    answer = tokens[answer_start]
  for i in range(answer_start+1, answer_end+1):
    if tokens[i][0:2] == "##":
      answer += tokens[i][2:]
    else:
      answer += " " + tokens[i]

  if answer.startswith("[CLS]"):
    answer = "Unable to find the answer to your question."
  print("\nPredicted Answer:\n{}".format(answer.capitalize()))
  return {"answer":answer}

@app.post('/upload')
async def uploadData(data:Request):
  obj=await data.json()
  text=obj["data"]
  chuncks=get_chunks(text,400)
  for document in chuncks:
    collection.insert_one({document:document})

Document1 = """what is the name of my university?"""

Document2 = """During their tenure in the University, students get exposure to an academic environment which is different from their future work environment, viz. industry, wherein they are expected to be placed. To get this exposure, all students (B.Tech., M.Tech.Integrated, B.Des. Industrial Design, BBA, B.Sc. Catering and Hotel Management) should undergo four weeks of industrial internship in a reputed industry in their respective discipline of study, any time after their first year of study. The industrial internship."""

print(cosineSimilarity(Document1,Document2))



