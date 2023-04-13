def fetchData(collection):
  dict={}
  docId=1
  cursor = collection.find({})
  for document in cursor:
          dict[docId]=document["document"]
          docId+=1
  return dict


  
