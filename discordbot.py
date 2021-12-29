##############   NLP - ML ##############
import pandas as pd

import random as rd

#Turkce stemmer ve autocorrect
from TurkishStemmer import TurkishStemmer
from turkishnlp import detector

#Nlp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob

#Verisetleri
from text import *

#Makine öğrenmesi modelleri
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

DISCORD_BOT_KEY = None

df = pd.DataFrame()

df['Text'] = dictionary_data.keys()
df['Intents'] = dictionary_data.values()

def clearingandconverting(text):
      
    text ="".join(text)                  # Virgüllerle ayrılmış listeyi join methodu ile bir cümle haline getirdim
    
    text=text.lower()                    # Buradan sonraki 4 satırd ise NLP methodlarını uygulayabilmek adına
                                         # bütün veriyi küçük harflere çevirdik ve içlerinden numerik 
                                         # verileri ve de sembolleri attık
    text=text.replace("[^\w\s]","") 
    text=text.replace("\d+","") 
    text=text.replace("\n","").replace("\r","") 
    return text

df['Text'] = df['Text'].apply(clearingandconverting)

stemmer = TurkishStemmer()

def split_into_lemmas(text):    # Stemma analiz methodunu tanımladık
    
    text = str(text).lower()   
    
    words = TextBlob(text).words
    
    return [stemmer.stem(word) for word in words]

X, y=df['Text'], df['Intents']

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.01, random_state=80)

vect=CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), analyzer=split_into_lemmas)

X_train_dtm=vect.fit_transform(X_train,y_train)

X_test_dtm=vect.transform(X_test)

b=MultinomialNB()

model=b.fit(X_train_dtm,y_train)

# Bu fonksiyonun amacı yanlış yazılan kelimeleri doğrusuna çevirmek
def correctfunc(text):
    
    obj = detector.TurkishNLP()

    obj.create_word_set()   # Gerekli fonksiyon

    lwords = obj.list_words(text)

    corrected_words = obj.auto_correct(lwords)
    
    corrected_string = " ".join(corrected_words)

    return corrected_string

def vectorizing(text):
    
    return vect.transform([text])

def vectorizingandclearing(text):
    
    return vectorizing(clearingandconverting((text)))

##############   NLP - ML ##############

import discord
from forex_python.converter import CurrencyRates

client = discord.Client()

def dolar():
    c = CurrencyRates()
    return c.get_rate("USD","TRY")

@client.event
async def on_ready():
    print("We have logged in as {0.user}".format(client))

@client.event
async def on_message(message):

    if str(message.author) != ("Chatbot#2319"):


        if message.content.startswith('$Chatbot'):

            await message.channel.send("Chatbot'a hoş geldiniz...")

        print((message.author))
                    
        if (b.predict(vectorizing(clearingandconverting((message.content)))) == 'slm'):
                        
            await message.channel.send("Chatbot Şükrü: "+ rd.choice(slm_list))

                        
        elif (b.predict(vectorizing(clearingandconverting((message.content)))) == 'veda'):

            await message.channel.send("Chatbot Şükrü: "+ rd.choice(veda_list))
                        
        elif (b.predict(vectorizing(clearingandconverting((message.content)))) == 'rica'):
                        
            await message.channel.send("Chatbot Şükrü: "+ rd.choice(rica_list))
                        
        elif (b.predict(vectorizing(clearingandconverting((message.content)))) == 'oyun'):
                        
            await message.channel.send("Chatbot Şükrü: "+ rd.choice(oyun_list))
                    
        elif (b.predict(vectorizing(clearingandconverting((message.content)))) == 'mizah'):
                        
            await message.channel.send("Chatbot Şükrü: "+ rd.choice(mizah_list))
                        
        elif (b.predict(vectorizing(clearingandconverting((message.content)))) == 'duygusal'):

            await message.channel.send(print("Chatbot Şükrü: "+ rd.choice(duygu_list)))

        elif (b.predict(vectorizing(clearingandconverting((message.content)))) == 'döviz'):

            await message.channel.send("Chatbot Şükrü: "+ rd.choice(doviz_list))

client.run(DISCORD_BOT_KEY)
