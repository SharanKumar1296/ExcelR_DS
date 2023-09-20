#==================================================================================
import pandas as pd 
import numpy as np 

df = pd.read_csv("Elon_musk.csv",encoding="Latin")
df

list(df)
df.drop(columns="Unnamed: 0",inplace=True)
df

df['Text'] = df.Text.map(lambda x : x.lower()) #converting to lowercase
df['Text']

df=[Text.strip() for Text in df.Text] #eliminating leading and trailing characters 

df=[Text for Text in df if Text] #removing empty strings

#Joining the list into one string
text = ' '.join(df)
text
type(text) #str
len(text)  #158702

#Generating WORDCLOUD
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def plot_cloud(wordcloud):
    plt.figure(figsize=(15, 30))
    plt.imshow(wordcloud) 
    plt.axis("off");
    
type(STOPWORDS)
stopwords = STOPWORDS
len(stopwords) #192

wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(text)
plot_cloud(wordcloud)

#Text pre-processing
import string 
no_punc_text = text.translate(str.maketrans('', '', string.punctuation)) 
no_punc_text

#Tokenization
from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
len(text_tokens) #21078
print(text_tokens[0:50])

#Removing stopwords
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
my_stop_words = stopwords.words('english')

no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]
len(no_stop_tokens) #14882
print(no_stop_tokens[0:40])

#Joining the words together
doc = ' '.join(my_stop_words)
doc

#Lemmatization
from nltk.stem import WordNetLemmatizer
Lemmatizer = WordNetLemmatizer()

lemmas = []
for token in doc.split():
    lemmas.append(Lemmatizer.lemmatize(token))

print(lemmas)
type(lemmas) #list

#Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lemmas)
X

pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=True).head(30)
pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=True)
print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names_out()[0:11])
print(vectorizer.get_feature_names_out()[50:100])
print(X.toarray()[50:100])

#Bigram
vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,1),max_features = 120)
bow_matrix_ngram =vectorizer_ngram_range.fit_transform(df)
bow_matrix_ngram
type(df)

print(vectorizer_ngram_range.get_feature_names_out())
print(bow_matrix_ngram.toarray())

print(vectorizer_ngram_range.get_feature_names_out())
w1 = list(vectorizer_ngram_range.get_feature_names_out())
type(w1)
w2 = ' '.join(w1)
w2
type(w2)

#Plot
wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=120,colormap='Set2',stopwords=stopwords).generate(w2)
plot_cloud(wordcloud)

#Trigram
vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(2,2),max_features = 100)
bow_matrix_ngram =vectorizer_ngram_range.fit_transform(df)
bow_matrix_ngram

print(vectorizer_ngram_range.get_feature_names_out())
print(bow_matrix_ngram.toarray())

w3 = list(vectorizer_ngram_range.get_feature_names_out())
w3
w4 = ' '.join(w3)
w4

#Plot
wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(w4)
plot_cloud(wordcloud)

#Sentiment Analysis
df=pd.read_csv('Elon_musk.csv',encoding='latin')
df.drop(columns='Unnamed: 0',inplace=True)
df
X=df['Text']

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
df['Sentiment'] = df['Text'].apply(lambda Text: analyzer.polarity_scores(Text)['compound'])

avg_sentiment = df['Sentiment'].mean()
positive_tweets = df[df['Sentiment'] > 0]
negative_tweets = df[df['Sentiment'] < 0]
neutral_tweets = df[df['Sentiment'] == 0]

print("Average Sentiment Score:", avg_sentiment)             #0.170
print("Number of Positive Tweets:", len(positive_tweets))    #883
print("Number of Negative Tweets:", len(negative_tweets))    #232
print("Number of Neutral Tweets:", len(neutral_tweets))      #884

#==================================================================================

