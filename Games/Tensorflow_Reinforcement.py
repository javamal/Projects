import gym
import tensorflow as tf
import numpy as np

# from https://gym.openai.com/docs/
# reinforcement learning practice
# the longer I can keep the pole in balance - higher the score
# want to create a model that can acheive an average score of 200
global_goal = 200

env = gym.make('CartPole-v0')
env.reset()
'''
set up training data
'''

def random_play():
    score_goal = 50
    total_trial = 10000
    total_moves = 500 #if it can survive for 300 moves I guess there's no harm in counting it as success  
    train_data = []
    all_score = []
    for trials in range(total_trial):
        env.reset()
        score_per_trial = 0
        data = []
        observations = []        
        for moves in range(total_moves):
            left_or_right = env.action_space.sample()
            
            obs, reward, done, info = env.step(left_or_right)            
            #obs = list of game descriptive figures
            #obs is an output of left_or_right
            #only makes sense I make a move based on observation
            #match current move with previous observation
            if moves > 0:
                data.append([observations, left_or_right])                      
            observations = obs           
            score_per_trial = score_per_trial + reward
            if done == True:
                break
            
        if score_per_trial > score_goal:
            for all_data in data:
                if all_data[1] == 1:
                    train_data.append([all_data[0], [0,1]]) #one hot for neural network
                elif all_data[1] == 0:
                    train_data.append([all_data[0], [1,0]])
        all_score.append(score_per_trial)
    #train_data = list of [observation input, move ouput]
    print("Total training data: ", len(train_data))
    print("Average score for random_play: ", np.mean(all_score))
    print("Maximum score for random_play: ", np.max(all_score))
    return(train_data)


#If I set score_goal too high, I need to substantially increase total_trial to get enough training data
#Average score for random_play is somewhere around 20
#random_play()

'''
set up neural network
'''
tf.reset_default_graph() 
#parameters
input_size = 4 #observations
#not sure how size was determined but it seems this is the general consensus in stackoverflow
#is there evidence to suggest this does better?
neuron_1 = 128
neuron_2 = 256
neuron_3 = 512
neuron_4 = 256
neuron_5 = 128 
output_size = 2 #move
total_epochs = 3
batch_size = 1

#define placeholders
x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

#define variables
layer_1_w = tf.Variable(tf.random_normal([input_size, neuron_1]))
layer_1_b = tf.Variable(tf.random_normal([1, neuron_1]))
layer_2_w = tf.Variable(tf.random_normal([neuron_1, neuron_2]))
layer_2_b = tf.Variable(tf.random_normal([1, neuron_2]))
layer_3_w = tf.Variable(tf.random_normal([neuron_2, neuron_3]))
layer_3_b = tf.Variable(tf.random_normal([1, neuron_3]))
layer_4_w = tf.Variable(tf.random_normal([neuron_3, neuron_4]))
layer_4_b = tf.Variable(tf.random_normal([1, neuron_4]))
layer_5_w = tf.Variable(tf.random_normal([neuron_4, neuron_5]))
layer_5_b = tf.Variable(tf.random_normal([1, neuron_5]))
output_layer_w = tf.Variable(tf.random_normal([neuron_5, output_size]))
output_layer_b = tf.Variable(tf.random_normal([1, output_size]))

saver = tf.train.Saver()
    
def feed_forward(data):    
    #using rectified linear units as activation functions
    #t(W * x) = t(x) * t(W)       
    layer_1_out = tf.nn.relu(tf.matmul(data, layer_1_w) + layer_1_b)
    layer_1_out = tf.nn.dropout(layer_1_out, 0.8)
    
    layer_2_out = tf.nn.relu(tf.matmul(layer_1_out, layer_2_w) + layer_2_b)
    layer_2_out = tf.nn.dropout(layer_2_out, 0.8)
    
    layer_3_out = tf.nn.relu(tf.matmul(layer_2_out, layer_3_w) + layer_3_b)
    layer_3_out = tf.nn.dropout(layer_3_out, 0.8)
    
    layer_4_out = tf.nn.relu(tf.matmul(layer_3_out, layer_4_w) + layer_4_b)
    layer_4_out = tf.nn.dropout(layer_4_out, 0.8)
    
    layer_5_out = tf.nn.relu(tf.matmul(layer_4_out, layer_5_w) + layer_5_b)
    layer_5_out = tf.nn.dropout(layer_5_out, 0.8)
    
    output_out = tf.matmul(layer_5_out, output_layer_w) + output_layer_b
    return(output_out)

def test_train(x):
    y_hat = feed_forward(x)
    cost = -tf.reduce_sum(y * tf.nn.log_softmax(y_hat))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    nn_data = random_play()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(total_epochs):
            loss = 0
            #nn_data = nn_data[np.random.permutation(len(nn_data))]
            x_train = [train[0] for train in nn_data]
            y_train = [train[1] for train in nn_data]    
            train_index = 0
            for batch in range(int(len(nn_data) / batch_size)):                
                batch_x = x_train[train_index : train_index + batch_size]
                batch_y = y_train[train_index : train_index + batch_size]
                op, c = sess.run([optimizer, cost], feed_dict = {x:batch_x, y:batch_y})
                loss = loss + c
                train_index = train_index + batch_size
            print("epoch: ", epoch, "total loss: ", loss)
        saver.save(sess, '/tmp/model.ckpt') #save this session and variables
        print("parameters saved")
        
test_train(x)

def nn_predict(obs):
    obs = obs.astype(np.float32).reshape([1,4])
       
    with tf.Session() as sess:
        saver.restore(sess, '/tmp/model.ckpt')
        y_hat = tf.nn.softmax(feed_forward(obs))
        result = sess.run(y_hat, feed_dict = {x:obs})[0]
        if result[1] >= result[0]:
            return(1)
        elif result[0] > result[1]:
            return(0)

'''
test to see if we can beat global goal
'''
def test():    
    test_trial = 100
    all_scores = []
    total_moves = 500
    for trials in range(test_trial):
        env.reset()
        observations = []
        score_per_trial = 0        
        for moves in range(total_moves):
            #env.render()
            if moves == 0:
                #first move made randomly
                left_or_right = env.action_space.sample()                
            else:                      
                #after the first move, model makes predictions
                #left_or_right = nn_predict(np.array(observations))  
                left_or_right = np.argmax(model.predict(observations.reshape(-1,len(observations),1))[0])
            obs, reward, done, info = env.step(left_or_right)            
            observations = obs #next observation determined by current move
            score_per_trial = score_per_trial + reward
            if done == True:
                break            
        print(trials, "'th trial ended")
        print(score_per_trial)
        all_scores.append(score_per_trial)
    print("Total trials played: ", test_trial)
    print("Average scores: ", np.average(all_scores))
    
    if np.average(all_scores) >= global_goal:
        print("pass")
        return(True)
    else:
        print("fail")
        return(False)

test()














