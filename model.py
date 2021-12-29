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

import pickle

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

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

while True:
    
    giris = input("Chatbot'a hoş geldiniz!,denemek istediğiniz cümleyi giriniz, kapatmak için q'ya basınız\n")
        
    sonuc = b.predict(vectorizingandclearing((giris)))

    if (giris =='q'):

        print("Kapanıyor...")

        break

    else:
        
        if (sonuc == 'slm'):
            
            print("Chatbot Şükrü: ", rd.choice(slm_list))
            
        elif (sonuc == 'veda'):
            
            print("Chatbot Şükrü: ", rd.choice(veda_list))
            
        elif (sonuc == 'rica'):
            
            print("Chatbot Şükrü: ", rd.choice(rica_list))
            
        elif (sonuc == 'oyun'):
            
            print("Chatbot Şükrü: ", rd.choice(oyun_list))
        
        elif (sonuc == 'mizah'):
            
            print("Chatbot Şükrü: ", rd.choice(mizah_list))
            
        elif (sonuc == 'duygusal'):
            
            print("Chatbot Şükrü: ", rd.choice(duygu_list))

        elif (sonuc == 'döviz'):

            print("Chatbot Şükrü: ", rd.choice(doviz_list))