import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from libs.utils import weight_variable, bias_variable
from tensorflow.examples.tutorials.mnist import input_data
import tensorlayer as tl
mnist = input_data.read_data_sets("data/", one_hot = True)

x_d = tf.placeholder(tf.float32, shape = [None, 784])
x_g = tf.placeholder(tf.float32, shape = [None, 128])

d_layer1=500
d_layer2=250

g_dim = 128
g_layer1=256
g_layer2=512


weights = {
    "w_d1" : weight_variable([784, d_layer1], "w_d1"),
    "w_d2" : weight_variable([d_layer1, d_layer2], "w_d2"),
    "w_d3" : weight_variable([d_layer2, 1], "w_d3"),


    "w_g1" : weight_variable([g_dim, g_layer1], "w_g1"),
    "w_g2" : weight_variable([g_layer1, g_layer2], "w_g2"),
    "w_g3" : weight_variable([g_layer2, 784], "w_g3")
}

biases = {
    "b_d1" : bias_variable([d_layer1], "b_d1"),
    "b_d2" : bias_variable([d_layer2], "b_d2"),
    "b_d3" : bias_variable([1], "b_d3"),

    "b_g1" : bias_variable([g_layer1], "b_g1"),
    "b_g2" : bias_variable([g_layer2], "b_g2"),
    "b_g3" : bias_variable([784], "b_g3")
}

var_d = [weights["w_d1"], weights["w_d2"], weights["w_d3"], biases["b_d1"], biases["b_d2"], biases["b_d3"]]
var_g = [weights["w_g1"], weights["w_g2"], weights["w_g3"], biases["b_g1"], biases["b_g2"], biases["b_g3"]]

def generator(z):
    h_g1 = tf.nn.relu(tf.add(tf.matmul(z, weights["w_g1"]), biases["b_g1"]))
    h_g2 = tf.nn.relu(tf.add(tf.matmul(h_g1, weights["w_g2"]),biases["b_g2"]))
    h_g3 = tf.nn.sigmoid(tf.add(tf.matmul(h_g2, weights["w_g3"]),biases["b_g3"]))
    return h_g3


def discriminator(x):
    h_d1 = tf.nn.relu(tf.add(tf.matmul(x, weights["w_d1"]), biases["b_d1"]))
    h_d2 = tf.nn.relu(tf.add(tf.matmul(h_d1, weights["w_d2"]), biases["b_d2"]))
    h_d3 = tf.nn.sigmoid(tf.add(tf.matmul(h_d2, weights["w_d3"]), biases["b_d3"]))
    return h_d3

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

g_sample = generator(x_g)
d_real= discriminator(x_d)
d_fake = discriminator(g_sample)

d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake)) 
g_loss = -tf.reduce_mean(tf.log(d_fake))
global_step = tf.Variable(0)

learning_rate_d = tf.train.exponential_decay(0.0005,decay_steps=1000,decay_rate=0.8,global_step=global_step)
d_optimizer = tf.train.AdamOptimizer(learning_rate_d).minimize(d_loss, var_list= var_d)
#d_optimizer=tf.train.GradientDescentOptimizer(learning_rate_d, use_locking=False).minimize(d_loss, var_list= var_d)


learning_rate_g = tf.train.exponential_decay(0.0001,decay_steps=1000, decay_rate=0.8,global_step=global_step)
g_optimizer = tf.train.AdamOptimizer(learning_rate_g).minimize(g_loss, var_list= var_g)
#g_optimizer=tf.train.GradientDescentOptimizer(learning_rate_g, use_locking=False).minimize(g_loss,var_list= var_g)

def plot(num,samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap = 'gray')

    plt.savefig('./pic/figure %d .png'%(num))
    #plt.show()

num=1
batch_size = 256
global_step=500001
sess = tf.InteractiveSession()
init_op = tf.initialize_all_variables()
saver = tf.train.Saver(max_to_keep=0)
sess.run(init_op)
for step in range(global_step):
    batch_x = mnist.train.next_batch(batch_size)[0]
    _, d_loss_train = sess.run([d_optimizer, d_loss], feed_dict = {x_d: batch_x, x_g: sample_Z(batch_size, g_dim)})
    _, g_loss_train = sess.run([g_optimizer, g_loss], feed_dict = {x_g: sample_Z(batch_size, g_dim)})
    

    if step % 1000 == 0:
            print("step %d, discriminator loss %.5f" % (step, d_loss_train)),
            print(" generator loss %.5f" % (g_loss_train))
    if step % 4000 == 0: 
            g_sample_plot = g_sample.eval(feed_dict = {x_g: sample_Z(16, g_dim)})
            plot(num,g_sample_plot)
            save_path = saver.save(sess, "./mod/model%d.ckpt"%(num))
            num+=1
