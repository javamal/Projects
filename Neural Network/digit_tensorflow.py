import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#parameters
input_size = 784
neuron_1 = 500
neuron_2 = 500
neuron_3 = 500
output_size = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def model(data):
    
    hidden_layer_1 = {"weights": tf.Variable(tf.random_normal([input_size, neuron_1])), \
                      "biases": tf.Variable(tf.random_normal([neuron_1]))}
                                                             
    hidden_layer_2 = {"weights": tf.Variable(tf.random_normal([neuron_1, neuron_2])), \
                      "biases": tf.Variable(tf.random_normal([neuron_2]))}
                                                             
    hidden_layer_3 = {"weights": tf.Variable(tf.random_normal([neuron_2, neuron_3])), \
                      "biases": tf.Variable(tf.random_normal([neuron_3]))}
    
    output_layer = {"weights": tf.Variable(tf.random_normal([neuron_3, output_size])), \
                    "biases": tf.Variable(tf.random_normal([output_size]))}

    out_1  = tf.add(tf.matmul(data, hidden_layer_1["weights"]), hidden_layer_1["biases"])
    activation_1 = tf.nn.relu(out_1)     

    out_2  = tf.add(tf.matmul(activation_1, hidden_layer_2["weights"]), hidden_layer_2["biases"])
    activation_2 = tf.nn.relu(out_2)

    out_3  = tf.add(tf.matmul(activation_2, hidden_layer_3["weights"]), hidden_layer_3["biases"])
    activation_3 = tf.nn.relu(out_3)   

    output_out  = tf.add(tf.matmul(activation_3, output_layer["weights"]), output_layer["biases"])
        
    return(output_out)       

    
def train_test_model(input_data):
    y_hat = model(input_data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = y))
    #cost = tf.reduce_sum(tf.square(y - y_hat))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    total_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())        
        for epoch in range(total_epochs):
            epoch_loss = 0; train_index = 0
            for epoch_iter in range(int(mnist.train.num_examples / batch_size)+1):
                epoch_x = mnist.train.images[train_index: train_index + batch_size]
                epoch_y = mnist.train.labels[train_index: train_index + batch_size]
                sess.run(optimizer, feed_dict = {x:epoch_x, y:epoch_y})
                c = sess.run(cost, feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss = epoch_loss + c
                train_index = train_index + batch_size
                print("trained up to:", train_index)
            print('Epoch', epoch, 'completed out of', total_epochs, 'loss:', epoch_loss)
            
        correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
            
train_test_model(x)
