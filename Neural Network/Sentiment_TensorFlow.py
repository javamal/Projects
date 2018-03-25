import nltk
import pickle
import os
import numpy as np
import tensorflow as tf

save_path = "C:\\Users\\donkey\\Desktop\\machine learning\\Neural Network\\sentiment\\ckpt"
'''
load test and train data
'''
with open("Sentiment_Data_BoW.pickle", "rb") as load_file:
    file = pickle.load(load_file) #label is mixed, no need to shuffle
    load_file.close()    

tf.reset_default_graph()
#parameters
input_size = len(file["train_x"][0])
neuron_1 = 500
neuron_2 = 500
neuron_3 = 500
output_size = len(file["train_y"][0])
total_epochs = 20
batch_size = 50

#define placeholders
x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float', [None, output_size])

#define variables
layer_1_w = tf.Variable(tf.random_normal([input_size, neuron_1]))
layer_1_b = tf.Variable(tf.random_normal([1, neuron_1]))
layer_2_w = tf.Variable(tf.random_normal([neuron_1, neuron_2]))
layer_2_b = tf.Variable(tf.random_normal([1, neuron_2]))
layer_3_w = tf.Variable(tf.random_normal([neuron_2, neuron_3]))
layer_3_b = tf.Variable(tf.random_normal([1, neuron_3]))
output_layer_w = tf.Variable(tf.random_normal([neuron_3, output_size]))
output_layer_b = tf.Variable(tf.random_normal([1, output_size]))

saver = tf.train.Saver()
    
def feed_forward(data):    
    #using rectified linear units as activation functions
    #t(W * x) = t(x) * t(W)
    layer_1_out = tf.matmul(data, layer_1_w) + layer_1_b
    layer_1_out = tf.nn.relu(layer_1_out)
    
    layer_2_out = tf.matmul(layer_1_out, layer_2_w) + layer_2_b
    layer_2_out = tf.nn.relu(layer_2_out)
    
    layer_3_out = tf.matmul(layer_2_out, layer_3_w) + layer_3_b
    layer_3_out = tf.nn.relu(layer_3_out)
    
    output_out = tf.matmul(layer_3_out, output_layer_w) + output_layer_b
    return(output_out)

def test_train(x):
    y_hat = feed_forward(x)
    cost = -tf.reduce_sum(y * tf.nn.log_softmax(y_hat))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(total_epochs):
            loss = 0
            shuffle_index = np.random.permutation(len(file["train_x"]))
            x_train = file["train_x"][shuffle_index]
            y_train = file["train_y"][shuffle_index]
            train_index = 0
            for batch in range(int(len(file["train_x"]) / batch_size)):                
                batch_x = x_train[train_index : train_index + batch_size]
                batch_y = y_train[train_index : train_index + batch_size]
                op, c = sess.run([optimizer, cost], feed_dict = {x:batch_x, y:batch_y})
                loss = loss + c
                train_index = train_index + batch_size
            print("epoch: ", epoch, "total loss: ", loss)
        saver.save(sess, os.path.join(save_path, 'model.ckpt')) #save this session and variables
        print("parameters saved")
        list_test = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1)) #by row
        prediction_rate = tf.reduce_mean(tf.cast(list_test, 'float'))
        print("prediction rate: ", sess.run(prediction_rate, feed_dict = {x: file["test_x"], y:file["test_y"]}))
    
#test_train(x) 

def convert_to_input(string, total_frame = file["frame"]):
    bag_of_words_item = np.zeros(len(total_frame))
    word_token = nltk.tokenize.word_tokenize(string.lower())        
    for word in word_token:
        lemmatized_word = nltk.stem.WordNetLemmatizer().lemmatize(word, "v")        
        if lemmatized_word in total_frame:              
            bag_of_words_item[total_frame.index(lemmatized_word)] = bag_of_words_item[total_frame.index(lemmatized_word)] + 1           
    bag_of_words_item = np.reshape(bag_of_words_item, (1, len(bag_of_words_item)))
    return(bag_of_words_item)
    
def instance_test(input_data):    
    with tf.Session() as sess:   
        saver.restore(sess, os.path.join(save_path, 'model.ckpt')) #load prameters to this session         
        y_hat = tf.nn.softmax(feed_forward(x))              
        result = sess.run(y_hat, feed_dict = {x:convert_to_input(input_data)})[0]        
        if result[0] > result[1]:
            print("positive")            
        elif result[1] > result[0]:
            print("negative")            
        else:
            print("neutal")                
            
test = [
        "I really want to buy another one because my first purchase was just too good",\
        "I was very disappointed to have my uranium confiscated at the airport. It was a gift for my son for his birthday. Also, I'm in prison now, so that's not good either.", \
        "comfortable fit and fast delivery 5 stars.", \
        "One drawback, when it was delivered the capsule had no bolt on the outside. But I’m handy, so I installed one.", \        
        "This was viewed by the wife she said she enjoyed it. I don't like to see her enjoying her self so one star", \
        "There are movies that are bad. There are movies that are so-bad-they're-good. And then there's Troll 2 -- a movie that's so bad that it defies comprehension",\
        "The jail was very nice and well clean. The cops were very friendly. The beds are also very comfortable.",\
        ]
        
for item in test:    
    instance_test(item)

'''
I really want to buy another one because my first purchase was just too good 
: positive
I was very disappointed to have my uranium confiscated at the airport. It was a gift for my son for his birthday. Also, I'm in prison now, so that's not good either. 
: negative
comfortable fit and fast delivery 5 stars. 
: positive
One drawback, when it was delivered the capsule had no bolt on the outside. But I’m handy, so I installed one. 
: negative
I love this cleanser!!! I purchased for the 1st time in August 2017 in within less than 3 weeks it cleared up my hormonal acne. 
: negative
This was viewed by the wife she said she enjoyed it. I don't like to see her enjoying her self so one star 
: positive
There are movies that are bad. There are movies that are so-bad-they're-good. And then there's Finding Dory -- a movie that's so bad that it defies comprehension 
: negative
The jail was very nice and well clean. The cops were very friendly. The beds are also very comfortable. 
: positive
'''
         
