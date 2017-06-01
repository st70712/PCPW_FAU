#2/28
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
from readfile_fau import readFile,separa_fau
from pylab import *

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot(num,samples):
    samples=np.concatenate((samples,np.zeros((samples.shape[0],16))),axis=1)
    fig = plt.figure(figsize=(1, 1))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0., hspace=0.)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(20, 20), cmap = 'gray')

    plt.savefig('./pic/figure %d .jpg'%(num))
    #plt.show()

filename_train = './data/fau_train_nor.arff'
filename_test =  './data/fau_test_nor.arff'
#train_data, train_label, train_size = readFile(filename_train)
train_data=np.load("./data/select_train.npy")
train_label=np.load("./data/select_label.npy")
print("訓練資料：",train_data.shape,train_label.shape)

fau_a,fau_e,fau_n,fau_p,fau_r=separa_fau(train_data,train_label)
print("Anger:",fau_a.shape,"Emphatic:",fau_e.shape,"Neutral:",fau_n.shape,"Positive:",fau_p.shape,"Rest:",fau_r.shape)

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

var_d = [weights["w_d1"], weights["w_d2"], weights["w_d3"],biases["b_d1"], biases["b_d2"], biases["b_d2"]]
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


epsilon=1e-9
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


##################################################開始訓練#####################################
train_data=fau_a #設定要用哪類情緒 



num=0
batch_size = 10
global_step=50001
#emotion=['Anger','Emphatic','Neutral','Positive','Rest']
#file_emo='./emotion_detail/'+emotion[0]#設定要存的資料夾位置'0:Anger,1:Emphatic,2:Neutral,3:Positive,4:Rest'

sess = tf.InteractiveSession()
init_op = tf.initialize_all_variables()
saver = tf.train.Saver({weights["w_g1"], weights["w_g2"], weights["w_g3"],biases["b_g1"], biases["b_g2"], biases["b_g3"]},max_to_keep=0)
sess.run(init_op)
if train_data.shape[0]==881:using_emo='Anger'
if train_data.shape[0]==2093:using_emo='Emphatic'
if train_data.shape[0]==5590:using_emo='Neutral'
if train_data.shape[0]==674:using_emo='Positive'
if train_data.shape[0]==721:using_emo='Rest'
print("使用情緒：",train_data.shape[0])

for step in range(global_step):
    offset = (step * batch_size) % (train_data.shape[0] - batch_size)
    batch_x = train_data[offset:(offset + batch_size), :]
    _, d_loss_train = sess.run([d_optimizer, d_loss], feed_dict = {x_d: batch_x, x_g: sample_Z(batch_size, g_dim)})
    _, g_loss_train = sess.run([g_optimizer, g_loss], feed_dict = {x_g: sample_Z(batch_size, g_dim)})
    if step % 1000 == 0:
            print("step %d, discriminator loss %.5f" % (step, d_loss_train)),
            print(" generator loss %.5f" % (g_loss_train))
            saver.save(sess, './mod'+"/gen_model%d.ckpt"%(num))           
    if step % 1000 == 0: 
            g_sample_plot = g_sample.eval(feed_dict = {x_g: sample_Z(1, g_dim)})
            plot(num,g_sample_plot)
            #saver.save(sess, file_emo+"/gen_model%d.ckpt"%(num))
    if step % 1000 == 0:num+=1
            
