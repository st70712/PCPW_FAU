import sklearn as sk
import numpy as np
import math
import tensorflow as tf
import sys
import time
import curses
from pylab import *
import tflearn as tl
from libs.utils import weight_variable, bias_variable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def readFile(filename, isSP=False) :
    f = open(filename, 'r')
    
    # read header
    line = f.readline()
    while line != '@data\n' :
        line = f.readline()
    
    line = f.readline()  #空白行
    
    #read data
    data = []
    label = []
    size = [0, 0, 0, 0, 0]
    
    while True:
        line = f.readline()
        if line == '' :  # EOF
            break
        
        #data processing
        temp = []
        for numstr in line.split(",") :
            if numstr :
                try :
                    numFl = float(numstr)
                    temp.append(numFl)
                except ValueError as e :
                    
                    if not isSP :
                        numstr = numstr[:-1]  #skip '\n'
                    else :
                        numstr = numstr[1]  #get class (supervector only)
                    
                    if numstr == 'Anger' or numstr == 'A' :##標為0
                        L = [1,0,0,0,0]
                        size[0] += 1
                    if numstr == 'Emphatic' or numstr == 'E' :##標為1
                        L = [0,1,0,0,0]
                        size[1] += 1
                    if numstr == 'Neutral' or numstr == 'N' :##標為2
                        L = [0,0,1,0,0]
                        size[2] += 1
                    if numstr == 'Positive' or numstr == 'P' :##標為3
                        L = [0,0,0,1,0]
                        size[3] += 1
                    if numstr == 'Rest' or numstr == 'R' :##標為4
                        L = [0,0,0,0,1]
                        size[4] += 1
                    
                    label.append(L)  # get label
        data.append(temp)  # get data
    
    f.close()
    
    data = np.asarray(data)
    label = np.asarray(label)
    #data, label = randomize(data, label)  #from Homework1 (Udacity)
    
    
    return data, label, size

def separa_fau(data,label):
    a=[]
    e=[]
    n=[]
    p=[]
    r=[]
    for i in range(label.shape[0]):
        if((np.argmax(label[i]))==0):a.append(data[i])
        if((np.argmax(label[i]))==1):e.append(data[i])
        if((np.argmax(label[i]))==2):n.append(data[i])
        if((np.argmax(label[i]))==3):p.append(data[i])
        if((np.argmax(label[i]))==4):r.append(data[i])
    a=np.asarray(a)
    e=np.asarray(e)
    n=np.asarray(n)
    p=np.asarray(p)
    r=np.asarray(r)
    return a,e,n,p,r            

filename_train = './data/fau_train_nor.arff'
filename_test =  './data/fau_test_nor.arff'
train_data, train_label, train_size = readFile(filename_train)
test_data, test_label, test_size = readFile(filename_test)
print("訓練資料：",train_data.shape,train_label.shape,train_size)
print("測試資料：",test_data.shape,test_label.shape)
fau_a,fau_e,fau_n,fau_p,fau_r=separa_fau(train_data,train_label)
print(fau_a.shape,fau_e.shape,fau_n.shape,fau_p.shape,fau_r.shape)

x_d = tf.placeholder(tf.float32, shape = [None, 384])
x_g = tf.placeholder(tf.float32, shape = [None, 100])

d_layer1=120
d_layer2=120

g_dim = 100
g_layer1=300
g_layer2=300



weights = {
    "w_d1" : weight_variable([384, d_layer1], "w_d1"),
    "w_d2" : weight_variable([d_layer1, d_layer2], "w_d2"),
    "w_d3" : weight_variable([d_layer2, 1], "w_d3"),


    "w_g1" : weight_variable([g_dim, g_layer1], "w_g1"),
    "w_g2" : weight_variable([g_layer1, g_layer2], "w_g2"),
    "w_g3" : weight_variable([g_layer2, 384], "w_g3")
}

biases = {
    "b_d1" : bias_variable([d_layer1], "b_d1"),
    "b_d2" : bias_variable([d_layer2], "b_d2"),
    "b_d3" : bias_variable([1], "b_d3"),

    "b_g1" : bias_variable([g_layer1], "b_g1"),
    "b_g2" : bias_variable([g_layer2], "b_g2"),
    "b_g3" : bias_variable([384], "b_g3")
}

var_d = [weights["w_d1"], weights["w_d2"], weights["w_d3"], biases["b_d1"], biases["b_d2"], biases["b_d3"]]
var_g = [weights["w_g1"], weights["w_g2"], weights["w_g3"], biases["b_g1"], biases["b_g2"], biases["b_g3"]]

#var_d = [weights["w_d1"], weights["w_d2"],biases["b_d1"], biases["b_d2"]]
#var_g = [weights["w_g1"], weights["w_g2"],biases["b_g1"], biases["b_g2"]]

def generator(z):
    h_g1 = tf.nn.relu(tf.add(tf.matmul(z, weights["w_g1"]), biases["b_g1"]))
    h_g2 = tf.nn.sigmoid(tf.add(tf.matmul(h_g1, weights["w_g2"]),biases["b_g2"]))
    h_g3 = tf.nn.sigmoid(tf.add(tf.matmul(h_g2, weights["w_g3"]),biases["b_g3"]))
    return h_g3


def discriminator(x):
    h_d1 = tf.nn.relu(tf.add(tf.matmul(x, weights["w_d1"]), biases["b_d1"]))
    h_d2 = tf.nn.sigmoid(tf.add(tf.matmul(h_d1, weights["w_d2"]), biases["b_d2"]))
    h_d3 = tf.nn.sigmoid(tf.add(tf.matmul(h_d2, weights["w_d3"]), biases["b_d3"]))
    return h_d3

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot(num,samples):
    samples=np.concatenate((samples,np.zeros((samples.shape[0],16))),axis=1)
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(20, 20), cmap = 'gray')

    plt.savefig('./pic/figure %d .png'%(num))
    #plt.show()

g_sample = generator(x_g)
d_real= discriminator(x_d)
d_fake = discriminator(g_sample)

d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))#希望d_real是1,d_fake是0.加負號後loss最小值為0 log(0.5)＝-0.7
g_loss = -tf.reduce_mean(tf.log(d_fake))#希望d_fake是1,loss最小值為0

global_step = tf.Variable(0)
# 只更新 discriminator
learning_rate_d = tf.train.exponential_decay(0.0005,decay_steps=10000,decay_rate=0.5,global_step=global_step)
d_optimizer = tf.train.AdamOptimizer(learning_rate_d).minimize(d_loss, var_list= var_d)
#d_optimizer=tf.train.GradientDescentOptimizer(learning_rate_d, use_locking=False).minimize(d_loss, var_list= var_d)

# 只更新 generator parameters
learning_rate_g = tf.train.exponential_decay(0.0001,decay_steps=10000, decay_rate=0.5,global_step=global_step)
g_optimizer = tf.train.AdamOptimizer(learning_rate_g).minimize(g_loss, var_list= var_g)
#g_optimizer=tf.train.GradientDescentOptimizer(learning_rate_g, use_locking=False).minimize(g_loss,var_list= var_g)

num=0
batch_size = 100
global_step=100001
sess = tf.InteractiveSession()
init_op = tf.initialize_all_variables()
saver = tf.train.Saver(max_to_keep=0)
sess.run(init_op)
train_data=fau_a #設定要用的情緒類別
for step in range(global_step):
    offset = (step * batch_size) % (train_data.shape[0] - batch_size)
    batch_x = train_data[offset:(offset + batch_size), :]
    _, d_loss_train = sess.run([d_optimizer, d_loss], feed_dict = {x_d: batch_x, x_g: sample_Z(batch_size, g_dim)})
    _, g_loss_train = sess.run([g_optimizer, g_loss], feed_dict = {x_g: sample_Z(batch_size, g_dim)})
    if step % 1000 == 0:
            print("step %d, discriminator loss %.5f" % (step, d_loss_train)),
            print(" generator loss %.5f" % (g_loss_train))
    if step % 1000 == 0: 
            g_sample_plot = g_sample.eval(feed_dict = {x_g: sample_Z(16, g_dim)})
            plot(num,g_sample_plot)
            save_path = saver.save(sess, "./mod/model%d.ckpt"%(num))
            num+=1
