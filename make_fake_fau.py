import numpy as np
import math
import tensorflow as tf
import sys
import time
import curses
from pylab import *
from libs.utils import weight_variable, bias_variable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from readfile_fau import readFile,separa_fau
from pylab import *

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

x_g = tf.placeholder(tf.float32, shape = [None, 100])
g_dim = 100
g_layer1=300
g_layer2=300

weights = {
    "w_g1" : weight_variable([g_dim, g_layer1], "w_g1"),
    "w_g2" : weight_variable([g_layer1, g_layer2], "w_g2"),
    "w_g3" : weight_variable([g_layer2, 384], "w_g3")
}
biases = {
    "b_g1" : bias_variable([g_layer1], "b_g1"),
    "b_g2" : bias_variable([g_layer2], "b_g2"),
    "b_g3" : bias_variable([384], "b_g3")
}

def generator(z):
    h_g1 = tf.nn.relu(tf.add(tf.matmul(z, weights["w_g1"]), biases["b_g1"]))
    h_g2 = tf.nn.sigmoid(tf.add(tf.matmul(h_g1, weights["w_g2"]),biases["b_g2"]))
    h_g3 = tf.nn.sigmoid(tf.add(tf.matmul(h_g2, weights["w_g3"]),biases["b_g3"]))
    return h_g3

number_of_data=10000
emotion=['Anger','Emphatic','Positive','Rest']
emo_list=[[],[],[],[],[]]
for i in range(4):
    file_emo='./emotion_detail/'+emotion[i]
    file_path=file_emo+'/gen_model50.ckpt'
    with tf.Session() as session:
        saver = tf.train.Saver({'Variable_3':weights['w_g1'],'Variable_4':weights['w_g2'],'Variable_5' :weights['w_g3'],'Variable_9':biases['b_g1'],'Variable_10' :biases['b_g2'],'Variable_11' :biases['b_g3']})
        saver.restore(session, file_path)
        g_sample = generator(x_g)
        new_fau_sample= g_sample.eval(feed_dict = {x_g:sample_Z(number_of_data,g_dim)})
        #plot(i, new_fau_sample)
        emo_list[i]=new_fau_sample
        session.close()

born_a=emo_list[0]
born_e=emo_list[1]
born_p=emo_list[2]
born_r=emo_list[3]
np.savetxt("./emotion_detail/fake_data/fake_a.txt",born_a)
np.savetxt("./emotion_detail/fake_data/fake_e.txt",born_e)
np.savetxt("./emotion_detail/fake_data/fake_n.txt",born_n)
np.savetxt("./emotion_detail/fake_data/fake_p.txt",born_p)
np.savetxt("./emotion_detail/fake_data/fake_r.txt",born_r)


