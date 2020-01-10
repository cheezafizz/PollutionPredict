#!/usr/bin/env python
# coding: utf-8

# In[1]:



from pandas import *
dat= read_csv("C:/Users/fjqld/Desktop/PollutionPredict-master/PollutionPredict-master/final_data.csv")
len(dat)	##3652


a=[]
b=[]
##1/1~2/28
for j in range(59):
    t=j
    for i in range(10):
        a.append(dat['미세먼지농도(㎍/㎥)'][t])
        if(i%4==3):
            t+=366
        else:
            t+=365
    b.append(a)
    a=[]


dat['미세먼지농도(㎍/㎥)'][59+365+365+366]


##3/1~12/31
for j in range(59,365):
    t=j
   # print(t)
    for i in range(10):
        a.append(dat['미세먼지농도(㎍/㎥)'][t])
        if(i%4==2):
            t+=366
        else:
            t+=365
    b.append(a)
    a=[]

##2/29 data##
feb_29 =[]
t = 365+365+365+59
feb_29.append(dat['미세먼지농도(㎍/㎥)'][t])
t+= 366+365+365+365
feb_29.append(dat['미세먼지농도(㎍/㎥)'][t])
feb_29


##make dataset
from numpy import *
day_365=nanmean(b,axis=1)
day_366=insert(day_365,59,37)
c=tile(day_365,3)		#2009~2011
c=append(c,day_366)		#2009~2012
c=tile(c,2)			#2009~2016
c=append(c,tile(day_365,2))	#2009~2018
#len(c)				#3652
smog_mean = DataFrame({'같은날미세먼지평균(㎍/㎥)':c})
new_data=  concat([dat, smog_mean.reset_index(drop=True)], axis=1) 


new_data.head()


# In[2]:


index = isnan(new_data['일강수량(mm)'])
new_data['일강수량(mm)'][index]=0


# In[3]:


new_data.head()


y_data = new_data['미세먼지농도(㎍/㎥)']
x_data = new_data[['평균기온(°C)','일강수량(mm)','평균 풍속(m/s)','최다풍향(16방위)','평균 상대습도(%)','평균 현지기압(hPa)','같은날미세먼지평균(㎍/㎥)']]
# mu = mean(new_data,axis=0)[['평균기온(°C)','일강수량(mm)','평균 풍속(m/s)','최다풍향(16방위)','평균 상대습도(%)','평균 현지기압(hPa)','같은날미세먼지평균(㎍/㎥)']]
# st = std(new_data,axis=0)[['평균기온(°C)','일강수량(mm)','평균 풍속(m/s)','최다풍향(16방위)','평균 상대습도(%)','평균 현지기압(hPa)','같은날미세먼지평균(㎍/㎥)']]
x_data = np.array(x_data)






import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()

X = tf.placeholder(tf.float32,shape=[1,7])
Y = tf.placeholder(tf.float32,shape=[1,1])


W1 = tf.Variable(tf.random_uniform([7, 7], -1., 1.))
b1 = tf.Variable(tf.zeros([7]))
 
# 신경망의 히든 레이어에 가중치 W1과 편향 b1을 적용합니다

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
 
 
W2 = tf.Variable(tf.random_normal([7, 7]))
b2 = tf.Variable(tf.zeros([7]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))

W3 = tf.Variable(tf.random_normal([7, 1]))
b3 = tf.Variable(tf.zeros([1]))
model = tf.add(tf.matmul(L2, W3), b3)


cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
 
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
 
for step in range(3652):
    sess.run(train_op, feed_dict={X: np.array([(np.float32(x_data[step]))]).T.T, Y: np.array([[y_data[step]]])})
 
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X:np.array([(np.float32(x_data[step]))]).T.T, Y: np.array([[y_data[step]]])}))
 


