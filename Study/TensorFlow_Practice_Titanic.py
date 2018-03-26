import pandas as pd
import numpy as np
import tensorflow as tf
import os
np.set_printoptions(threshold=np.nan)

##############################################
#set train and test data
##############################################
data = pd.read_excel("C:\\Users\\donkey\\Desktop\\titanic.xls")
df = data.ix[:,["survived", "pclass", "sex", "age", "sibsp", "ticket", "fare"]].dropna(axis = 0, how = "any").reset_index()
categorical = ["sex", "ticket"]

def assign_label(x):
    for feature in categorical:
        x.ix[:,feature] = x.ix[:,feature].astype(str)
        copy = np.array(x.ix[:,feature])
        unique = np.unique(copy)
        label = [t for t in range(len(unique))]
        for i in range(len(x)):
            copy[i] = label[np.where(unique ==copy[i])[0][0]]
        x.ix[:,feature] = copy
    return(x)
    
def one_hot(x):
    return_array = []
    for i in range(len(x)):
        if x[i] == 0:
            return_array.append([0,1])
        else:
            return_array.append([1,0])   
    return(return_array)
            
dependent = np.array(one_hot(df.ix[:,"survived"]))
independent = np.array(assign_label(df.ix[:,["pclass", "sex", "age", "sibsp", "ticket", "fare"]]))

train_y = dependent[0:int(len(dependent)*0.9)]
train_x = independent[0:int(len(dependent)*0.9)]

test_y = dependent[int(len(dependent)*0.9):]
test_x = independent[int(len(dependent)*0.9):]

##############################################
#neural network
##############################################
input_size = len(independent[0])
neuron_1 = 100
neuron_2 = 100
output_size = 2
batch_size = 94
total_epochs = 500

x = tf.placeholder('float', [None, input_size])   
y = tf.placeholder('float', [None, output_size])

def feed_forward(data):
    layer_1 = {"weights": tf.Variable(tf.random_normal([input_size, neuron_1])),\
               "bias": tf.Variable(tf.random_normal([1, neuron_1]))}
    layer_2 = {"weights": tf.Variable(tf.random_normal([neuron_1, neuron_2])),\
               "bias": tf.Variable(tf.random_normal([1, neuron_2]))}
    output_layer = {"weights": tf.Variable(tf.random_normal([neuron_2, output_size])),\
                    "bias": tf.Variable(tf.random_normal([1, output_size]))}
    
    out_1 = tf.matmul(data, layer_1["weights"]) + layer_1["bias"]
    out_1 = tf.nn.relu(out_1)
       
    out_2 = tf.matmul(out_1, layer_2["weights"]) + layer_2["bias"]
    out_2 = tf.nn.relu(out_2)
    
    output_out = tf.matmul(out_2, output_layer["weights"]) + output_layer["bias"]
    
    return(output_out)
    
   
def test_train_nn(x, rate = 0):
    y_hat = feed_forward(x)        
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = y)) #[1,0] or [0,1]
    optimize = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(total_epochs):
            total_loss = 0; batch_index = 0
            shuffle_index = np.random.permutation(np.arange(len(train_x)))
            train_x_shuffle = train_x[shuffle_index]
            train_y_shuffle = train_y[shuffle_index]
            for batch in range(int(len(train_x) / batch_size)):
                op, loss = sess.run([optimize, cost], \
                                    feed_dict = {x:train_x_shuffle[batch_index:batch_index + batch_size],
                                                 y:train_y_shuffle[batch_index:batch_index + batch_size]})
                total_loss = total_loss + loss                
                batch_index = batch_index + batch_size
            print(epoch, " epochs completed. Total cost: ", total_loss)
            
        list_test = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1)) #by row
        prediction_rate = tf.reduce_mean(tf.cast(list_test, 'float'))
        print("prediction rate: ", sess.run(prediction_rate, feed_dict = {x: test_x, y:test_y}))
            
test_train_nn(x)
