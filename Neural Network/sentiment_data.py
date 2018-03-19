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
data clean -> create frame -> create bag of words
'''
  
def word_frame(data, file_name = "wordframe.pickle"):
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
    
#word_frame(sentiment)
    
    
for w in range(len(word_dictionary)):
    if list(word_dictionary.values())[w] > lower:            
        word_frame_total.append(list(word_dictionary.keys())[w]) #filter out words   

    
def bag_of_words(data):
    '''
    bag_of_words() takes in a dictionary of raw text and label and returns a dictionary of
    input data in vector count format and label in one-hot vector format
    '''
    total_frame = word_frame(data)
    x_raw = data["x"]; y_raw = data["y"]
    bag = []
    for sentences in x_raw:        
        bag_of_words_item = np.zeros(len(total_frame))
        word_token = nltk.tokenize.word_tokenize(sentences.lower()) 
        '''
        use sparse matrix
        '''
        for word in word_token:
            if word in total_frame:            
                bag_of_words_item[total_frame.index(word)] = bag_of_words_item[total_frame.index(word)] + 1
        if(sum(bag_of_words_item)==0):
            print(sentences)
        bag.append(bag_of_words_item)
    return({"bag_of_words":bag, "label":y_raw})
    
def train_test(data, test_portion = 0.1):    
    data = bag_of_words(data)
    random_index = np.random.permutation(len(data["bag_of_words"]))
    data["bag_of_words"] = np.array(data["bag_of_words"])[random_index]
    data["label"] = np.array(data["label"])[random_index]
    
    index = int(len(data["bag_of_words"]) * test_portion)
    train_x = data["bag_of_words"][-index:]
    test_x = data["bag_of_words"][:index]
    train_y = data["label"][-index:]
    test_y = data["label"][:index]
    return({"train_x":train_x, "train_y":train_y, "test_x":test_x, "test_y":test_y})  

def save_sentiment_bow(data):
    save_data = train_test(data)
    with open("bagofwords.pickle", "wb") as save_file:
        pickle.dump(save_data, save_file)
        save_file.close()
    return(True)
    
save_sentiment_bow(sentiment)

