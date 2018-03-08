#same as neural_network_from_scratch.py 
#using tensorflow
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
total_epochs = 10
batch_size = 100

#define placeholders
x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float', [None, output_size])

#set model and feed forward
def feed_forward(data):
    layer_1 = {"weights":tf.Variable(tf.random_normal([input_size, neuron_1])),
               "bias":tf.Variable(tf.random_normal([1, neuron_1]))}

    layer_2 = {"weights":tf.Variable(tf.random_normal([neuron_1, neuron_2])),
               "bias":tf.Variable(tf.random_normal([1, neuron_2]))}

    layer_3 = {"weights":tf.Variable(tf.random_normal([neuron_2, neuron_3])),
               "bias":tf.Variable(tf.random_normal([1, neuron_3]))}

    output_layer = {"weights":tf.Variable(tf.random_normal([neuron_3, output_size])),
                    "bias":tf.Variable(tf.random_normal([1, output_size]))}
    
    #t(W * x) = t(x) * t(W)
    layer_1_out = tf.matmul(data, layer_1["weights"]) + layer_1["bias"]
    layer_1_out = tf.nn.relu(layer_1_out)
    
    layer_2_out = tf.matmul(layer_1_out, layer_2["weights"]) + layer_2["bias"]
    layer_2_out = tf.nn.relu(layer_2_out)
    
    layer_3_out = tf.matmul(layer_2_out, layer_3["weights"]) + layer_3["bias"]
    layer_3_out = tf.nn.relu(layer_3_out)
    
    output_out = tf.matmul(layer_3_out, output_layer["weights"]) + output_layer["bias"]
    
    return(output_out)

def test_train(x):
    y_hat = feed_forward(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_hat))
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
            
        list_test = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1)) #by row
        prediction_rate = tf.reduce_mean(tf.cast(list_test, 'float'))
        print("prediction rate: ", sess.run(prediction_rate, feed_dict = {x: mnist.test.images, y:mnist.test.labels}))

test_train(x)
