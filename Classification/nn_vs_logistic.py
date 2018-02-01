from neural_network import Neural_Network
import requests, re, pandas as pd, numpy as np
import sklearn.linear_model

url="http://www.stat.ufl.edu/~winner/data/stature_hand_foot.dat"

def url_to_text(url):
    try:
        source=requests.get(url)
    except:
        return("download failed")
        
    df_text=source.text
    
    array=re.split('\n', df_text)
    df=[]
    for i in array:
        row=re.sub(" {1,}"," ",i)
        row=re.sub("^ ","",row)
        row=re.split(" ",row)
        df=df+[row] 
    df=pd.DataFrame(df)
    df=df.dropna(axis=0)    
    return(df)

df = url_to_text(url)
 

df.columns=['index','y','x1','x2','x3']
index = np.arange(len(df)); np.random.shuffle(index)
df = df.ix[index,:].astype(float)

for i in ["x1","x2","x3"]:
    mean = np.mean(df.ix[:,i])
    sd = np.std(df.ix[:,i])
    df.ix[:,i] = (df.ix[:,i] - mean)/sd


y=np.array(df.ix[:,'y']).astype(float); y = y - 1
x=np.array(df.ix[:,['x1','x2','x3']])

size = 120
train_x = x[0:size]; test_x = x[size:len(x)]
train_y = y[0:size]; test_y = y[size:len(y)]  


'''
logistic reg
'''
log = sklearn.linear_model.LogisticRegression()
log.fit(train_x,train_y)
log_predict = log.predict(test_x)
sum(log_predict==test_y)/len(test_y)



train_y_array = []
for i in train_y:
    if i == 1:
        train_y_array = train_y_array + [np.array([1,0])]
    else:
        train_y_array = train_y_array + [np.array([0,1])]
train = [ [from_x,from_y]  for from_x, from_y in zip(train_x,train_y_array)]

'''
neural network
'''
a = Neural_Network([3,3,2])
#SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None)
a.SGD(train, 10000, 1, 0.0001)

predict = [a.feed_forward(i[0])["out"][-1] for i in train]
def test(predict,actual):
    ac = []
    for i in range(len(predict)):
        if predict[i][0] < predict[i][1]:
            val = np.array([0,1])
        else:
            val = np.array([1,0])
        print(val)
        if sum(val == actual[i][1])==2:
            ac = ac + [True]
        else:
            ac = ac + [False]            
    return(ac)    
b = test(predict,train)

print("CV MSE logistic regression: "+str(sum(log_predict==test_y)/len(test_y)))
print("CV MSE neural network: "+str(sum(b)/len(b)))

