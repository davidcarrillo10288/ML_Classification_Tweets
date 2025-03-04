from nltk.sentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math

# Cargamos el modelo de sarcasmo
analyzer_sarcasmo = SentimentIntensityAnalyzer()
analyzer = SentimentIntensityAnalyzer()


## función para sarcasmo
def detect_sarcasm(text):
    sentiment = analyzer_sarcasmo.polarity_scores(text)

    # Detecta posibles sarcasmos cuando hay un contraste entre palabras positivas y negativas
    if sentiment['neg'] > 0.4 and sentiment['pos'] > 0.2:
        return 1  # Sarcasmo detectado
    return 0  # No sarcástico


## función para entropía
def calculate_entropy(text):
    prob = [text.count(c) / len(text) for c in set(text)]
    entropy = -sum([p * math.log(p) for p in prob])
    return entropy


## función para neg_words
def neg(text):
  text = str(text)
  scores = analyzer.polarity_scores(text)
  return round(scores['neg']*len(text.split()))


## función para neu_words
def neu(text):
  text = str(text)
  scores = analyzer.polarity_scores(text)
  return round(scores['neu']*len(text.split()))


## función para pos_words
def pos(text):
  text = str(text)
  scores = analyzer.polarity_scores(text)
  return round(scores['pos']*len(text.split()))