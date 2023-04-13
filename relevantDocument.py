from cosine import cosineSimilarity

class pair:
  def __init__(self,document,cosineValue):
    self.document=document
    self.cosineValue=cosineValue

def findRelevantDocument(dict,query):
  relevantDocs=[]
  dataArr=[]
  for key,value in dict.items():
    cosineSimilarityValue=cosineSimilarity(query,value)
    dataArr.append(pair(value,cosineSimilarityValue))
  dataArr.sort(key=lambda x:x.cosineValue)
  for i in range(len(dataArr)-1,max(0,len(dataArr)-10),-1):
    relevantDocs.append(dataArr[i])

  return relevantDocs



