import os
import collections
import nltk
import re
import pickle
import numpy as np

path = "C:\\Users\\donkey\\Desktop\\machine learning\\Data\\sentiment labelled sentences\\sentiment labelled sentences\\sentiment.txt"

def data_clean(path):
    with open(os.path.join(path)) as file:
        x = []; y = []
        text = list(file)
        for i in text:
            try:
                label_raw = re.findall("\t([0|1])", i)[0]
            except:
                print("error in: " + i)
            if label_raw == "1":
                y.append([1,0])
            elif label_raw == "0":
                y.append([0,1])                
            text_raw = re.sub("\t([0|1])\n", "", i)
            x.append(text_raw)
        return({"x":x, "y":y})
        
def word_frame(path, lower = 10):
    counter = []
    text = data_clean(path)["x"]
    for i in text:
        text_token = nltk.tokenize.word_tokenize(i.lower())
        counter = counter + text_token       
    stop = nltk.corpus.stopwords.words('english') + ["!", "?", "-", ",", ")", "("]
    counter = [word for word in counter if not word in stop]
    clean = [nltk.stem.WordNetLemmatizer().lemmatize(word_wo_stop) for word_wo_stop in counter]
    word_frame_total = []
    word_dictionary = collections.Counter(clean)
    for w in range(len(word_dictionary)):
        if list(word_dictionary.values())[w] > lower:            
            word_frame_total.append(list(word_dictionary.keys())[w])    
    return(word_frame_total)
    
def bag_of_words(path):
    total_frame = word_frame(path)
    raw = data_clean(path); x_raw = raw["x"]; y_raw = raw["y"]
    bag = []
    for sentences in x_raw:
        bag_of_words_item = np.zeros(len(total_frame))
        word_token = nltk.tokenize.word_tokenize(sentences.lower())
        for word in word_token:
            if word in total_frame:
                bag_of_words_item[total_frame.index(word)] = bag_of_words_item[total_frame.index(word)] + 1
        bag.append(bag_of_words_item)
    return({"bag_of_words":bag, "label":y_raw})

def train_test(path, test_portion = 0.1):
    data = bag_of_words(path)
    index = int(len(data["bag_of_words"]) * test_portion)
    train_x = data["bag_of_words"][-index:]
    test_x = data["bag_of_words"][:index]
    train_y = data["label"][-index:]
    test_y = data["label"][:index]
    return({"train_x":np.array(train_x), "train_y":np.array(train_y), "test_x":np.array(test_x), "test_y":np.array(test_y)})  


with open("bagofwords.pickle", "wb") as save_file:
    save_data = train_test(path)
    pickle.dump(save_data, save_file)
    save_file.close()
