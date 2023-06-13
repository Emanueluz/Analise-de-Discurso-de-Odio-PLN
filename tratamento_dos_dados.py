import pandas as pd
import numpy as np
import csv
import spacy
import re
import matplotlib.pyplot as plt
import emoji

from unicodedata import normalize
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate



def preprocessar_tweets(tweets):
    nlp = spacy.load('pt_core_news_sm')
    repetion_pattern = re.compile(r'(.)\1\1+')
    new_tweets = []
    with tqdm(total=len(tweets), colour='green', desc='Processando') as pbar:
      for tweet in tweets[1:]:
          tweet[0] = emoji.demojize(tweet[0], language='pt')
          tweet[0] = tweet[0].replace('_', ' ')
          tweet[0] = normalize('NFKD', tweet[0]).encode('ASCII', 'ignore').decode('ASCII')
          tweet[0] = repetion_pattern.sub(r'\1', tweet[0])
          tweet[0] = re.sub(r'@\w+', ' ', tweet[0])
          tweet[0] = re.sub(r'\s\s+', ' ', tweet[0])
          doc = nlp(tweet[0])
          tokens = [t.lemma_.lower() for t in doc if t.pos_ != 'PUNCT' and not t.is_stop and len(t.lemma_) > 1]
          tweet[0]= ' '.join(tokens)
          tweet[0]= ' '.join(tweet[0].strip())
          tweet[1]=int(tweet[1])
          tweet[2]=int(tweet[2])
          tweet[3]=tweet[3].replace(",", ".", 1)
          tweet[3]=float(tweet[3])
          print(tweet)
          new_tweets.append(tweet)
          pbar.update(1)
           
    return new_tweets



def ler_csv(arquivo):
    matriz = []
    
    with open(arquivo, 'r') as csv_file:
        leitor = csv.reader(csv_file)
        
        for linha in leitor:
            matriz.append(linha)
    
    return matriz

arquivo_csv = 'dados.csv'  # Substitua pelo nome do seu arquivo CSV

matriz = ler_csv('/home/emanuel/Área de Trabalho/tt_odio/HateBR-main/dataset/HateBR.csv')
 

data=preprocessar_tweets(matriz)

 

dataset= pd.read_csv('/home/emanuel/Área de Trabalho/tt_odio/HateBR-main/dataset/HateBR.csv')
 #dataset=preprocessar_tweets(dataset)
 
# instagram_comments,     offensive_language,     offensiveness_levels,    hate_speech

 

'''
n_rows = 1000

comentario_train = dataset['train']['text']
comentario_validation = dataset['validation']['text']
comentario_test = dataset['test']['text']

labels_train = dataset['train']['label']
labels_validation = dataset['validation']['label']
labels_test = dataset['test']['label']


if n_rows > 0:
  comentario_train = comentario_train[:n_rows]
  comentario_validation = comentario_validation[:n_rows]
  comentario_test = comentario_test[:n_rows]

  labels_train = labels_train[:n_rows]
  labels_validation = labels_validation[:n_rows]
  labels_test = labels_test[:n_rows]


print(f'\nTrain: {len(comentario_train)}')
print(f'Validation: {len(comentario_validation)}')
print(f'Test: {len(comentario_test)}')

print(f'\n\nLabels Distribution Train: {Counter(labels_train)}')
print(f'\nLabels Distribution Validation: {Counter(labels_validation)}')
print(f'\nLabels Distribution Test: {Counter(labels_test)}')'''