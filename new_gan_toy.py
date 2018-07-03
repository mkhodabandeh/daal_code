import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf
from toy import load_data

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
#import tflib.plot

MODE = 'dcgan' # dcgan, wgan, or wgan-gp
DIM = 2# Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 100000 # How many generator iterations to train for 
OUTPUT_DIM = 2# Number of pixels in MNIST (28*28)
MODEL_PATH='./toy_models/'
NOISE_DIM = 4
DISC_MAP_NUMBER = 100
lib.print_model_settings(locals().copy())
fig = plt.figure()
ax = fig.add_subplot(111)

def save_plot(data, filepath):
    ax.clear()
    plt.scatter(data[:,0], data[:,1], marker='o' )
    plt.savefig(filepath)

def save_plot2(data, orig_data, disc_map, filepath):
    ax.clear()
    #plt.scatter(disc_map[:,0], disc_map[:,1], marker='.', alpha=0.3, c=disc_map[:,2]) 
    plt.scatter(disc_map[:,0], disc_map[:,1], c=disc_map[:,2]) 
    plt.scatter(orig_data[:,0], orig_data[:,1], marker='+', color='blue', alpha=0.1) 
    plt.scatter(data[:,0], data[:,1], marker='o', color='green' )
    plt.savefig(filepath)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, NOISE_DIM])

    output = lib.ops.linear.Linear('Generator.Input', NOISE_DIM, 2, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    #output = tf.nn.relu(output)
    #output = lib.ops.linear.Linear('Generator.1', 2, 2, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):

    #output = LeakyReLULayer('Discriminator.1', 2, 2, inputs)
    output = lib.ops.linear.Linear('Discriminator.Input', 2, 1, inputs)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    #output = tf.nn.tanh(output)
    #output = lib.ops.linear.Linear('Discriminator.Output', 4, 1, output)

    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

disc_map_data= tf.placeholder(tf.float32, shape=[DISC_MAP_NUMBER**2, OUTPUT_DIM])
disc_map = Discriminator(disc_map_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.zeros_like(disc_fake)
    ))
    disc_cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real, 
        labels=tf.ones_like(disc_real)
    ))
    disc_cost += disc_cost1
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-3, 
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-5, 
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(1024, NOISE_DIM)).astype('float32'))
fixed_noise_samples = Generator(NOISE_DIM, noise=fixed_noise)


# Train loop
import pdb
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    test_data, train_gan = load_data() 
    #min_train = np.min(train_gan[0], axis=1)
    #max_train = np.max(train_gan[0], axis=1) - min_train
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        total = DISC_MAP_NUMBER * 1.0 
	all_data = np.concatenate((samples, train_gan[0]), axis=0)
         
        min_x = np.min(all_data[:,0])
        min_y = np.min(all_data[:,1])
        max_x = np.max(all_data[:,0])
        max_y = np.max(all_data[:,1])
        #x_range = np.arange(min_x, max_x, (max_x-min_x)/total)[:DISC_MAP_NUMBER].reshape((int(total), 1))
        #y_range = np.arange(min_y, max_y, (max_y-min_y)/total)[:DISC_MAP_NUMBER].reshape((int(total), 1))
        x_range = np.arange(-10, 10, 20/total)[:DISC_MAP_NUMBER].reshape((int(total), 1))
        y_range = np.arange(-10, 10, 20/total)[:DISC_MAP_NUMBER].reshape((int(total), 1))
        meshx, meshy = np.meshgrid(x_range, y_range)
        data_range = np.zeros((DISC_MAP_NUMBER**2, 2))
        for i in range(DISC_MAP_NUMBER):
            for j in range(DISC_MAP_NUMBER):
                data_range[i*DISC_MAP_NUMBER+j] = np.array([meshx[i,j], meshy[i,j]])
        #data_range = np.concatenate((x_range, y_range), axis=1)
        _disc_map = session.run(
            disc_map,
            feed_dict={disc_map_data: data_range}
        )
        #samples[:,0] = (samples[:,0]*max_train[0] + min_train[0])
        #samples[:,1] = (samples[:,1]*max_train[1] + min_train[1])
	_disc_map = _disc_map.reshape((DISC_MAP_NUMBER**2, 1))
        _disc_map = np.concatenate((data_range, _disc_map), axis=1)
	#pdb.set_trace()
	save_plot2(samples, train_gan[0], _disc_map, 'samples/samples_{}.png'.format(frame))
        #lib.save_images.save_images(
        #    samples.reshape((4, 28, 28)), 
        #    'samples/samples_{}.png'.format(frame)
        #)
    def inf_train_gen():
        data = train_gan[0] 
	save_plot(data, 'data.png')
        print test_data[0].shape
        print data.shape
        #data[:,0] = (data[:,0]-min_train[0])/max_train[0]
        #data[:,1] = (data[:,1]-min_train[1])/max_train[1]
        while True:
            np.random.shuffle(data)
            for i in range(0, data.shape[0], BATCH_SIZE): 
                yield data[i:i+BATCH_SIZE, ...]
    gen = inf_train_gen()
    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _, _gan_cost = session.run([gen_train_op, gen_cost])
            
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _, _gan_cost,_disc_cost1 = session.run(
                [disc_cost, disc_train_op, gen_cost, disc_cost1],
                feed_dict={real_data: _data}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        #lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            #dev_disc_costs = []
            #for images,_ in dev_gen():
            #    _dev_disc_cost = session.run(
            #        disc_cost, 
            #        feed_dict={real_data: images}
            #    )
            #    dev_disc_costs.append(_dev_disc_cost)
            #lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, _data, )
            print('{3}\treal+fake {0}\treal {1}\tgen {2}'.format( _disc_cost,_disc_cost1, _gan_cost,iteration ))

        # Write logs every 100 iters
        #if (iteration < 5) or (iteration % 100 == 99):
        #    lib.plot.flush()

        #lib.plot.tick()
