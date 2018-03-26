#activation function: relu / softmax
#loss function: cross entropy
#optimizer: adaptive moment estimation
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

input_size = 784
neuron_1 = 100
output_size = 10
total_epochs = 100
batch_size = 100

x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])


def feed_forward(x):
    layer_1 = {"weights":tf.Variable(tf.random_normal([input_size, neuron_1])),\
               "bias":tf.Variable(tf.random_normal([1, neuron_1]))}                                      
                                                   
    output_layer = {"weights":tf.Variable(tf.random_normal([neuron_1, output_size])),\
                    "bias":tf.Variable(tf.random_normal([1, output_size]))}
    
    #saving variables in dictionary might not be the best idea for saving variables? Not sure how this works
                                                                                       
    layer_1_out = tf.matmul(x, layer_1["weights"]) + layer_1["bias"]
    layer_1_out = tf.nn.relu(layer_1_out)
    
    output_out = tf.matmul(layer_1_out, output_layer["weights"]) + output_layer["bias"]
    return(output_out)

def test_train(x):
    y_hat = feed_forward(x)
    cost = -tf.reduce_sum(y * tf.nn.log_softmax(y_hat))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(total_epochs):
            loss = 0
            shuffle_index = np.random.permutation(mnist.train.num_examples)
            x_train = mnist.train.images[shuffle_index]
            y_train = mnist.train.labels[shuffle_index]
            train_index = 0
            for batch in range(int(mnist.train.num_examples / batch_size)):                
                batch_x = x_train[train_index : train_index + batch_size]
                batch_y = y_train[train_index : train_index + batch_size]
                op, c = sess.run([optimizer, cost], feed_dict = {x:batch_x, y:batch_y})
                loss = loss + c                
                train_index = train_index + batch_size
            print("epoch: ", epoch, "total loss: ", loss)   
            
        
        test = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1)) #y_hat is list of softmax probabilites
        prediction_rate = tf.reduce_mean(tf.cast(test, tf.float32))
        rate = sess.run(prediction_rate, feed_dict = {x:mnist.test.images, y:mnist.test.labels})
        print("prediction rate: ", rate)
          
test_train(x)                       
    
                        
