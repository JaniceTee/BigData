# import the MongoClient class from the library
from pymongo import MongoClient

# import Natural Language Toolkit library
import nltk

# import nltk stopwords list
from nltk.corpus import stopwords

# import pandas
import pandas as pd

# import regular expression library
import re

# import heap queue algorithm library
import heapq as heapq

# create a client instance of the MongoDB database class
client = MongoClient("mongodb://localhost:27017/")

# create an instance of database
db = client["movie_review"]

# access a MongoDB collection
collection = db["datamining_project"]

# make a dataframe of only "text" field
data = pd.DataFrame(collection.find({}, {"text": 1}), columns=["text"])

# take a random sample from text dataframe, n is the sample size
data_sample = data.sample(n=500, random_state=1)

# Making a paragraph
paragraph = ""
for review in data_sample["text"]:
    paragraph += review + "\n"
print(paragraph)

# Tokenize sentences: break the texts in the paragraph to a list of sentences.
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

# Creating word histogram
word2count = {}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1

# Selecting best 100 features
freq_words = heapq.nlargest(100, word2count, key=word2count.get)

# create a term frequency table; the table contains the most used words in reviews
term_frequency_table = pd.DataFrame(word2count.items(), columns=["Term", "Frequency"])

# sort the table by frequency and show top n=50 terms
sorted_table = term_frequency_table.sort_values(by="Frequency", ascending=False).head(n=50).to_string(index=False)
print('\nTerm Frequency Table')
print(sorted_table)