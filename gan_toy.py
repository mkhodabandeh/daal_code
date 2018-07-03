import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf
import tflearn
from toy import load_data
import tensorflow.contrib.slim as slim
import tflib as lib
import tflib.plot

MODE = 'dcgan' # dcgan, wgan, or wgan-gp
DIM = 2 # Model dimensionality
BATCH_SIZE = 50 # Batch size
TEST_BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for 
OUTPUT_DIM = 2 # Number of pixels in MNIST (28*28)
SNAPSHOT_INTERVAL=100
MODEL_PATH='./toy_models/'
lib.print_model_settings(locals().copy())


fig = plt.figure()
ax = fig.add_subplot(111)
def save_plot(data, filepath):
    ax.clear()
    plt.scatter(data[:,0], data[:,1], marker='o' )
    plt.savefig(filepath)
    
def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def Generator(n_samples, noise=None, reuse=False):
	if noise is None:
	    noise = tf.random_normal([n_samples, 2])
	out = tflearn.fully_connected(noise, 2, name='Generator.1',activation='relu'  )
	out = tf.reshape(out, [-1, 2],name='Generator.Output_reshaped')
	return out

def Discriminator(inputs, reuse=False):
    with tf.variable_scope('foo', reuse=reuse) as scope:
        out = tflearn.fully_connected(inputs, 1, name='Discriminator.1',activation='relu')
        out= tf.reshape(out, [-1],name='Discriminator.Output_reshaped')
        return out, tf.sigmoid(out, name='Discriminator.sigmoid')

if __name__=='__main__':
    real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
    fake_data = Generator(BATCH_SIZE)
    
    disc_real, disc_real_sigmoid = Discriminator(real_data)
    disc_fake, disc_fake_sigmoid = Discriminator(fake_data, reuse=True)
    gen_params = [ v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'Generator' in v.name]
    disc_params = [ v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'Discriminator' in v.name]
    
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.ones_like(disc_fake)
    ))
    
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real, 
        labels=tf.ones_like(disc_real)
    ))
    disc_cost /= 2.
    
    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)
    
    clip_disc_weights = None
    
    # For saving samples
    fixed_noise = tf.constant(np.random.normal(size=(1024, 4)).astype('float32'))
    fixed_noise_samples = Generator(1024, noise=fixed_noise)
    test_data, train_gan = load_data() 
    #min_train = np.min(train_gan[0], axis=1)
    #max_train = np.max(train_gan[0], axis=1) - min_train
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        #samples[:,0] = (samples[:,0]*max_train[0] + min_train[0])
        #samples[:,1] = (samples[:,1]*max_train[1] + min_train[1])
	save_plot(samples, 'samples/samples_{}.png'.format(frame))
        #lib.save_images.save_images(
        #    samples.reshape((4, 28, 28)), 
        #    'samples/samples_{}.png'.format(frame)
        #)
    
    # Dataset iterator
    #train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
    #train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, TEST_BATCH_SIZE, portions=[0, 0.2], labels=mlabels)
    def inf_train_gen():
        data = train_gan[0] 
	save_plot(data, 'data.png')
        #data[:,0] = (data[:,0]-min_train[0])/max_train[0]
        #data[:,1] = (data[:,1]-min_train[1])/max_train[1]
        while True:
            np.random.shuffle(data)
            for i in range(0, data.shape[0], BATCH_SIZE): 
                yield data[i:i+BATCH_SIZE, ...]
    
    saver = tf.train.Saver(max_to_keep=20)
    if not os.path.isdir(MODEL_PATH):
        os.system('mkdir '+MODEL_PATH)
    
    # Train loop
    with tf.Session() as session:
    
        session.run(tf.initialize_all_variables())
    
        gen = inf_train_gen()
    
        _disc_cost = 0
        for iteration in xrange(ITERS):
            if iteration%SNAPSHOT_INTERVAL == 0:
                saver.save(session, MODEL_PATH+'/my-model')
                print "iteration", iteration
            start_time = time.time()
    
            if iteration > 0:
                _ = session.run(gen_train_op)
    
            if MODE == 'dcgan':
                disc_iters = 1
            else:
                disc_iters = CRITIC_ITERS
	    #print 'disc_iters', disc_iters
            for i in xrange(disc_iters):
                #print 'next'
                _data = gen.next()
                #print 'data', _data
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data: _data}
                )
                #print 'done'
                if clip_disc_weights is not None:
                    _ = session.run(clip_disc_weights)
	    #'train disc cost', _disc_cost 
                        # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 99:
                generate_image(iteration, _data)
    
            # Write logs every 100 iters
