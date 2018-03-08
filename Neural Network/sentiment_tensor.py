import pickle
import numpy as np
import tensorflow as tf

with open("bagofwords.pickle", "br") as load_file:
    file = pickle.load(load_file) #label is mixed, no need to shuffle
    load_file.close()
  
#parameters
input_size = len(file["train_x"][0])   
neuron_1 = 50
neuron_2 = 50
output_size = len(file["train_y"][0])
total_epochs = 100
batch_size = 100

#define placeholders
x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float', [None, output_size])

#set model and feed forward
def feed_forward(data):
    layer_1 = {"weights":tf.Variable(tf.random_normal([input_size, neuron_1])),
               "bias":tf.Variable(tf.random_normal([1, neuron_1]))}
    layer_2 = {"weights":tf.Variable(tf.random_normal([neuron_1, neuron_2])),
               "bias":tf.Variable(tf.random_normal([1, neuron_1]))}
    output_layer = {"weights":tf.Variable(tf.random_normal([neuron_2, output_size])),
                    "bias":tf.Variable(tf.random_normal([1, output_size]))}
    
    #t(W * x) = t(x) * t(W)
    layer_1_out = tf.matmul(data, layer_1["weights"]) + layer_1["bias"]
    layer_1_out = tf.nn.relu(layer_1_out)
    
    layer_2_out = tf.matmul(layer_1_out, layer_2["weights"]) + layer_2["bias"]
    layer_2_out = tf.nn.relu(layer_2_out)
    
    output_out = tf.matmul(layer_2_out, output_layer["weights"])    
    return(output_out)

def test_train(x):
    y_hat = feed_forward(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_hat))
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
            
        list_test = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1)) #by row
        prediction_rate = tf.reduce_mean(tf.cast(list_test, 'float'))
        print("prediction rate: ", sess.run(prediction_rate, feed_dict = {x: file["test_x"], y:file["test_y"]}))

test_train(x)
    
