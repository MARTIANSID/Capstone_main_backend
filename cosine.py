import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def cosineSimilarity(query,document):
  count_vect = CountVectorizer()

  corpus = [query,document]

  X_train_counts = count_vect.fit_transform(corpus)

  from sklearn.feature_extraction.text import TfidfVectorizer
  vectorizer = TfidfVectorizer()

  trsfm=vectorizer.fit_transform(corpus)
  from sklearn.metrics.pairwise import cosine_similarity

  return cosine_similarity(trsfm[0:1], trsfm)[0][1]



