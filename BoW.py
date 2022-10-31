# import the MongoClient class from the library
from pymongo import MongoClient

# import CountVectorizer to convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer

# import Natural Language Toolkit library
import nltk

# import nltk stopwords list
from nltk.corpus import stopwords

# import pandas
import pandas as pd

# import regular expression library
import re as re

# create a client instance of the MongoDB cloud database class
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

# Bag of Words Model in NLP
bow_vectorizer = CountVectorizer()

# turn a collection of text documents into numerical feature vectors
bow_vect = bow_vectorizer.fit_transform(dataset)
print('\nBoW Vectors')
print(bow_vect)

# Bag of Words Term-Document Matrix
bow_tdm = pd.DataFrame(bow_vect.toarray().transpose(), index=bow_vectorizer.get_feature_names_out())
# print Bag of Words Term-Document Matrix of 100 random reviews
print("\nBag of Words Term-Document Matrix")
print(bow_tdm)
