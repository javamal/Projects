import os
import collections
import nltk
import re
import pickle
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.nan)

with open("sentiment_data.pickle", "rb") as save_file:
    print("loading raw text and label")
    sentiment = pickle.load(save_file)
    save_file.close()

def filtered(lower, file = "wordframe.pickle"):    
    '''
    filtered selects words that were most common from total_frame
    '''
    with open(file, "rb") as read_file:
        frame = pickle.load(read_file)
        read_file.close()
    filtered_frame = []
    for i in range(len(frame)):
        if list(frame.values())[i] > lower:
            filtered_frame.append(list(frame.keys())[i])
    return(filtered_frame)
    
def bag_of_words(lower, data = sentiment):
    '''
    bag_of_words() takes in sentiment data and filtered total frame
    input data in vector count format and label in one-hot vector format
    '''
    total_frame = filtered(lower)
    x_raw = data["x"]; y_raw = data["y"]
    bag = []
    for sentences in x_raw:        
        bag_of_words_item = np.zeros(len(total_frame))
        word_token = nltk.tokenize.word_tokenize(sentences.lower())        
        for word in word_token:
            if word in total_frame:            
                bag_of_words_item[total_frame.index(word)] = bag_of_words_item[total_frame.index(word)] + 1
        if(sum(bag_of_words_item)==0):
            print(sentences)
        bag.append(bag_of_words_item)
    return({"bag_of_words":bag, "label":y_raw, "text":x_raw, "frame":total_frame})
    
def train_test(lower, test_portion):    
    data = bag_of_words(lower)
    random_index = np.random.permutation(len(data["bag_of_words"]))    
    data["bag_of_words"] = np.array(data["bag_of_words"])[random_index]
    data["label"] = np.array(data["label"])[random_index]    
    index = int(len(data["bag_of_words"]) * test_portion)
    train_x = data["bag_of_words"][-index:]
    test_x = data["bag_of_words"][:index]
    train_y = data["label"][-index:]
    test_y = data["label"][:index]
    return({"train_x":train_x, "train_y":train_y, "test_x":test_x, "test_y":test_y})  

def save_sentiment_bow(lower, test_portion, data = sentiment, file = "bagofwords.pickle"):
    save_data = train_test(lower, test_portion)
    with open(file, "wb") as save_file:
        pickle.dump(save_data, save_file)
        save_file.close()
        print("saved as: file")
    return(True)
    
#save_sentiment_bow(lower = 75, test_portion = 0.1)    
    
    
a = bag_of_words(50)

i = 20200
a["text"][i]    
np.array(a["frame"])[np.where(a["bag_of_words"][i]==1)[0]]
a["label"]     
    
    
