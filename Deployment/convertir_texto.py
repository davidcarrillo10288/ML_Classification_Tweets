from new_variables import detect_sarcasm, calculate_entropy, neg, neu, pos
import string
import pandas as pd
import numpy as np
from clean_text_func import clean_text_func
import textblob
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import nltk
nltk.download('vader_lexicon')

## creamos una función general para poder convertir el texto
def convertir_texto(text):
  # global clean_text, neg, neu, pos
  df = pd.DataFrame({'text':[str(text)]})
  df['clean_text'] = df['text'].apply(lambda x: clean_text_func(x))
  df['without_stopwords'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
  df['num_palabras'] = df['clean_text'].apply(lambda x: len(str(x).split()))
  # df['num_palabras_unicas'] = df['clean_text'].apply(lambda x: len(set(str(x).split())))
  df['polarity'] = df['without_stopwords'].apply(lambda x: TextBlob(x).sentiment.polarity)
  df['subjectivity'] = df['without_stopwords'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
  df['num_stopwords'] = df['clean_text'].apply(lambda x: len([word for word in x.split() if word in stopwords.words('english')]))
  df['neg_words'] = df['without_stopwords'].apply(neg)
  df['neu_words'] = df['without_stopwords'].apply(neu)
  df['pos_words'] = df['without_stopwords'].apply(pos)
  df['num_exclamaciones'] = df['text'].apply(lambda x: x.count('!')).apply(lambda x: 0 if x == 0 else 1)
  df['num_interrogaciones'] = df['text'].apply(lambda x: x.count('?')).apply(lambda x: 0 if x == 0 else 1)
  df['num_puntuacion'] = df['text'].apply(lambda x: len(set([c for c in x if c in string.punctuation])))
  df['capital_word_density'] = df['text'].apply(lambda x: sum(1 for word in x.split() if word.isupper()) / len(x.split()) if x.split() else 0)
  df['num_repeticiones'] = df['clean_text'].apply(lambda x: len([word for word in x.split() if x.split().count(word) > 1]))
  df['sarcasmo'] = df['text'].apply(detect_sarcasm)
  df['entropia'] = df['clean_text'].apply(calculate_entropy)

  return df

def tranformation(df):
  df_num = df.drop(columns=['text', 'clean_text', 'without_stopwords'])
  ## aplicando transformación logarítmica num_neg, pos, neu, num_stopwords y entropía
  df_num['neg_words'] = np.log1p(df_num['neg_words'])
  df_num['neu_words'] = np.log1p(df_num['neu_words'])
  df_num['pos_words'] = np.log1p(df_num['pos_words'])
  df_num['num_stopwords'] = np.log1p(df_num['num_stopwords'])
  df_num['entropia'] = np.log1p(df_num['entropia'])
  return df_num