from nltk.sentiment import vader


sia = vader.SentimentIntensityAnalyzer()
sia.polarity_scores("this place is disgusting")
print(sia['compoud'])