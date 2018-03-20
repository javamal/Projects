import os
import collections
import nltk
import re
import pickle
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.nan)

'''
three sets of sentiment data organized in different format
'''
reviews = "C:\\Users\\donkey\\Desktop\\machine learning\\Data\\Sentiment\\reviews.txt"
def review_sentiment(review_path):
    with open(review_path, "r") as file:
        x = []; y = []
        text = list(file)
        for i in text:
            try:
                label_raw = re.findall("\t([0|1])", i)[0]
            except:
                print("error in: " + i)
            if label_raw == "1":
                y.append([1,0,0])
            elif label_raw == "0":
                y.append([0,0,1])                
            text_raw = re.sub("\t([0|1])\n", "", i)
            x.append(text_raw)
        return({"x":x, "y":y})
    
twitter = "C:\\Users\\donkey\\Desktop\\machine learning\\Data\\Sentiment\\twitter.txt"
def twitter_sentiment(twitter_path):
    with open(twitter_path, "r", encoding = "utf-8") as file:
        x = []; y = []
        text = list(file)
        for i in text:
            try:
                label_raw = re.findall("([0|1])\t", i)[0]
            except:
                print("error in: " + i)
            if label_raw == "1":
                y.append([1,0,0])
            elif label_raw == "0":
                y.append([0,0,1])                
            text_raw = re.sub("([0|1])\t", "", i)
            text_raw = re.sub("\n", "", text_raw)
            x.append(text_raw)
        return({"x":x, "y":y})
     
twitter_2 = "C:\\Users\\donkey\\Desktop\\machine learning\\Data\\Sentiment\\text_emotion.csv"
def twitter_sentiment_2(twitter_2_path):
    df = pd.read_csv(twitter_2_path)
    x = []; y = []
    for i in range(len(df)):
        if (df.ix[i,"sentiment"] == "happiness") or (df.ix[i,"sentiment"] == "fun") or (df.ix[i,"sentiment"] == "love") or (df.ix[i,"sentiment"] == "relief"):
            y.append([1,0,0])
        elif df.ix[i,"sentiment"] == "neutral":
            y.append([0,1,0])
        else:
            y.append([0,0,1])                
    x = list(df.ix[:,"content"])        
    return({"x":x, "y":y})

'''
combined data in dictionary format
data will be a dictionary of keys x and y. x: raw text and y: one-hot label     
'''

sentiment = {"x":review_sentiment(reviews)["x"] + twitter_sentiment(twitter)["x"] + twitter_sentiment_2(twitter_2)["x"], \
             "y":review_sentiment(reviews)["y"] + twitter_sentiment(twitter)["y"] + twitter_sentiment_2(twitter_2)["y"]} 

'''
with ("sentiment_data.pickle", "wb") as save_file:
    pickle.dump(sentiment, save_file)
    save_file.close()
'''
    
def word_frame(data = sentiment, file_name = "wordframe.pickle"):
    '''
    word_frame() takes in a dictionary of raw text and returns a template for the input layer
    each input data will be transformed to a count vector based on this template
    '''
    counter = []; text = data["x"] 
    for i in text:                                         
        text_token = nltk.tokenize.word_tokenize(i.lower())
        counter = counter + text_token #all words      
        stop = nltk.corpus.stopwords.words('english') + [",", ".", "-", "!", "?", ")", "(", ":", "'", ":", ";", "%", "&"]
        counter = [word for word in counter if not word in stop] #filter stop words from all words
    clean = [nltk.stem.WordNetLemmatizer().lemmatize(word_wo_stop) for word_wo_stop in counter]    
    word_dictionary = collections.Counter(clean) #create count list from all words
    with open(file_name, "wb") as save_file:
        pickle.dump(word_dictionary, save_file)
        save_file.close()
        print("word frame dictionary saved as pickle file") 
    return(word_dictionary) 


    
    


       
