#same as neural_network_from_scratch.py 
#using tensorflow

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#parameters
input_size = 784
neuron_1 = 30
output_size = 10
total_epochs = 50
batch_size = 10

#define placeholders
x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float', [None, output_size])

#set model and feed forward
def feed_forward(data):
    layer_1 = {"weights":tf.Variable(tf.random_normal([input_size, neuron_1])),
               "bias":tf.Variable(tf.random_normal([1, neuron_1]))}
    output_layer = {"weights":tf.Variable(tf.random_normal([neuron_1, output_size])),
                    "bias":tf.Variable(tf.random_normal([1, output_size]))}
    
    #t(W * x) = t(x) * t(W)
    layer_1_out = tf.matmul(data, layer_1["weights"])
    layer_1_out_relu = tf.nn.sigmoid(layer_1_out)
    
    output_out = tf.matmul(layer_1_out_relu, output_layer["weights"])
    
    return(output_out)

def test_train(x, rate):
    y_hat = feed_forward(x)
    cost = tf.reduce_mean(tf.square(y_hat - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = rate).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(total_epochs):
            loss = 0
            for batch in range(int(mnist.train.num_examples / batch_size)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                op, c = sess.run([optimizer, cost], feed_dict = {x:batch_x, y:batch_y})
                loss = loss + c
            print("epoch: ", epoch, "total loss: ", loss)
            
        list_test = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1)) #by row
        prediction_rate = tf.reduce_mean(tf.cast(list_test, 'float'))
        print("prediction rate: ", sess.run(prediction_rate, feed_dict = {x: mnist.test.images, y:mnist.test.labels}))

test_train(x, 0.5)
