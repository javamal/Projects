import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#parameters
input_size = 784
neuron_1 = 30
output_size = 10
total_epochs = 10
batch_size = 100

#define placeholders
x = tf.placeholder('float', [input_size, 1])
y = tf.placeholder('float')

def feed_forward(data):    
    layer_1 = {"weights":tf.Variable(tf.random_normal([neuron_1, input_size])), \
               "bias":tf.Variable(tf.random_normal([neuron_1, 1]))}
    output_layer = {"weights":tf.Variable(tf.random_normal([output_size, neuron_1])), \
                    "bias":tf.Variable(tf.random_normal([output_size, 1]))}
    #Note t(x dot y) = t(y) dot t(x)
    out_1 = tf.matmul(layer_1["weights"], data) + layer_1["bias"]
    out_1 = tf.nn.relu(out_1)    
    output_1 = tf.matmul(output_layer["weights"], out_1) + output_layer["bias"]
    return(output_1)
    
def train_test_model(data, rate):
    y_hat = feed_forward(data)
    cost = tf.reduce_sum(tf.square(y_hat - y))
    back_prop = tf.train.GradientDescentOptimizer(rate).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(total_epochs):
            loss = 0
            shuffle_index = np.random.permutation(mnist.train.num_examples)
            x_images = mnist.train.images[shuffle_index]
            y_labels = mnist.train.labels[shuffle_index]
            for x_row, y_row in zip(x_images, y_labels):
                x_train = np.reshape(x_row, (input_size, 1))
                y_train = np.reshape(y_row, (output_size, 1))
                sess.run(back_prop, feed_dict = {x:x_train, y:y_train})
                loss = loss + sess.run(cost, feed_dict = {x:x_train, y:y_train})
            print("epochs: ", epoch, " Total loss: ", loss)
        
        test = []
        for x_row_test, y_row_test in zip(mnist.test.images[0:50], mnist.test.labels[0:50]):
            x_test = np.reshape(x_row_test, (input_size, 1))
            y_test = np.reshape(y_row_test, (output_size, 1))
            test.append(sess.run(tf.equal(tf.argmax(y_hat), tf.argmax(y_test)), feed_dict = {x:x_test, y:y_test}))
            print(sess.run(tf.argmax(y_hat), feed_dict = {x:x_test, y:y_test}))
        prediction_rate =  sum(test) / len(test)
        print(prediction_rate)                
            
train_test_model(x, 0.00000005)    
