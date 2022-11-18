# import the MongoClient class from the library
from pymongo import MongoClient

# import TfidVectorizer to convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer

# import Natural Language Toolkit library
import nltk

# import nltk stopwords list
from nltk.corpus import stopwords

# import pandas
import pandas as pd

import numpy as np

# import regular expression library
import re

# create a client instance of the MongoDB database class
client = MongoClient("mongodb://localhost:27017/")

# create an instance of database
db = client["movie_review"]

# access a MongoDB collection
collection = db["datamining_project"]

# make a dataframe of only "text" field
data = pd.DataFrame(collection.find({}, {"text": 1}), columns=["text"])

# take a random sample from text dataframe, n is the sample size
data_sample = data.sample(n=100, random_state=1)

# Making a paragraph
paragraph = ""
for review in data_sample["text"]:
    paragraph += review + "\n"
print(paragraph)

# Tokenize sentences
dataset = nltk.sent_tokenize(paragraph)
for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W', ' ', dataset[i])
    dataset[i] = re.sub(r'\s+', ' ', dataset[i])
print('\nTokenized Paragraph')
print(dataset)

# Removing stopwords and numbers
for i in range(len(dataset)):
    words = nltk.word_tokenize(dataset[i])
    words = [word for word in words if word not in stopwords.words('english') and not word.isdigit()]
    dataset[i] = ' '.join(words)
print('\nTokenized Paragraph with Stop words removed')
print(dataset)

# Term Frequency-Inverse Document Frequency Model in NLP
tfidf_vectorizer = TfidfVectorizer()

# turn a collection of text documents into numerical feature vectors
tfidf_vect = tfidf_vectorizer.fit_transform(dataset)
print('\nTF-IDF Vectors')
print(tfidf_vect)


# Term Frequency-Inverse Document Frequency Matrix
tfidf_tdm = pd.DataFrame(tfidf_vect.toarray().transpose(), index=tfidf_vectorizer.get_feature_names_out())

# print Term Frequency-Inverse Document Frequency Matrix
print("\nTerm Frequency-Inverse Document Frequency Matrix")
print(tfidf_tdm)



