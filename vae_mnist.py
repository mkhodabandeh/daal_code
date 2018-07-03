import os, sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from tensorflow.examples.tutorials.mnist import input_data
batch_size = 50 #We have to define the batch size with the current version of TensorFlow


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def get_mnist():
    mnist = input_data.read_data_sets('MNIST_data')
    return (mnist.train.images, mnist.train.labels), (mnist.test.images, mnist.test.labels)

def to_one_hot(Y, n=None):
    if not n:
	    n = int(np.max(Y))+1
    y = np.zeros((Y.shape[0], n))
    goods = (Y<n)
    y[goods, Y[goods].astype(int)] = 1
    return y

def get_val(num=1000, onehot=True, labels=range(5)):
    (train_x, train_y), (test_x, test_y) = get_mnist()
    X = np.zeros((len(labels)*num, train_x.shape[1]))
    Y = np.zeros((len(labels)*num, ))
    for i,label in enumerate(labels):
       indices = np.nonzero(test_y==label)[0]
       X[i*num:(i+1)*num, ...] = test_x[indices[:num], ...]
       Y[i*num:(i+1)*num, ...] = test_y[indices[:num], ...]
    if onehot:
        Y = to_one_hot(Y)
    return X, Y
 
def get_clean(num=200, onehot=False, reverse=False, labels=range(5)):
    (train_x, train_y), (test_x, test_y) = get_mnist()
     
    X = np.zeros((len(labels)*num, train_x.shape[1]))
    Y = np.zeros((len(labels)*num, ))
    for i,label in enumerate(labels):
       indices = np.nonzero(train_y==label)[0]
       if reverse:
		indices = indices[::-1]
       X[i*num:(i+1)*num, ...] = train_x[indices[:num], ...]
       Y[i*num:(i+1)*num, ...] = train_y[indices[:num], ...]
    if onehot:
        Y = to_one_hot(Y)
    return X, Y

def get_noisy_data(num=5000, outlier_percent=1, onehot=False):
    ### Good data ###
    X, Y = get_clean(num=num, labels=range(5), onehot=onehot)
    ### Noisy data ###
    noisy_num = int(outlier_percent*num)
    X_noisy, Y_noisy = get_clean(num=noisy_num, labels=range(5, 10), reverse=True, onehot=onehot)
    print 'Number of outliers:', X_noisy.shape[0], 'Total number of data:',  X_noisy.shape[0]+X.shape[0]

    X = np.concatenate((X, X_noisy), axis=0) 
    Y = np.concatenate((Y, Y_noisy), axis=0) 
    if onehot:
        Y = to_one_hot(Y, n=2)
    return X, Y

def run_all(session, network, inp, data, batch_size=128):
    arr = np.zeros((data.shape[0],)) 
    for i in range(data.shape[0]/batch_size):
        temp = session.run(network, feed_dict={inp:data[i*batch_size:(i+1)*batch_size, ...]})
        arr[i*batch_size:(i+1)*batch_size] = temp
    i = data.shape[0]/batch_size
    if i*batch_size != data.shape[0]:
        #temp = session.run(network, feed_dict={inp:data[i*batch_size:]})
        temp = session.run(network, feed_dict={inp:data[-batch_size:]})
        tt=len(data)-i*batch_size
        arr[i*batch_size:] = temp[-tt:]
    return arr

def get_grid(x_minmax, y_minmax, NUM):
        x_range = np.arange(x_minmax[0], x_minmax[1], 
			abs(x_minmax[1]-x_minmax[0])/NUM)
        x_range = x_range[:NUM].reshape((int(NUM), 1))
        y_range = np.arange(y_minmax[0], y_minmax[1], 
			abs(y_minmax[1]-y_minmax[0])/NUM)
	y_range = y_range[:NUM].reshape((int(NUM), 1))
        meshx, meshy = np.meshgrid(x_range, y_range)
        data_range = np.zeros((NUM**2, 2))
        for i in range(NUM):
            for j in range(NUM):
                data_range[i*NUM+j] = np.array([meshx[i,j], meshy[i,j]])
        return data_range

def nb(batch_size, non_crossing=True):
    z_true = np.random.uniform(0,1,batch_size)
    r = np.power(z_true, 0.5)
    phi = 0.25 * np.pi * z_true
    x1 = r*np.cos(phi)
    x2 = r*np.sin(phi)
    
    # Sampling form a Gaussian
    x1 = np.random.normal(x1, 0.10* np.power(z_true,2), batch_size)
    x2 = np.random.normal(x2, 0.10* np.power(z_true,2), batch_size)
        
    # Bringing data in the right form
    X = np.transpose(np.reshape((x1,x2), (2, batch_size)))
    X = np.asarray(X, dtype='float32')
    return X

def next_batch(batch_size, non_crossing=True):
    d1 = nb(batch_size/2)
    d2 = nb(batch_size/2)
    d2[:,1] *= -1
    d2[:,1] += 0.2
    d2[:,0] = 0.55-d2[:,0]    
    return np.concatenate((d1, d2), axis=0)

def my_next_batch(batch_size, data):
    data = data.copy()
    while True:
        np.random.shuffle(data)
        for i in range(data.shape[0]/batch_size):
            yield data[i*batch_size:(i+1)*batch_size]

# Functions to get variables
import tensorflow as tf
print("Tensor Flow version {}".format(tf.__version__))

def weights(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#### encoder network ####
def encoder(x, n_z=2):

	#n_z = 1 #Dimension of the latent space
	# Input
	#x = tf.placeholder("float32", shape=[None, 2]) #Batchsize x Number of Pixels
	n_hidden_1 = 500 
	n_hidden_2 = 501 
	n_hidden_3 = 503 

	# First hidden layer
	W_fc1 = weights([x.shape[1].value, n_hidden_1])
	b_fc1 = bias([n_hidden_1])
	h_1   = tf.nn.softplus(tf.matmul(x, W_fc1) + b_fc1)

	# Second hidden layer
	W_fc2 = weights([n_hidden_1, n_hidden_2]) 
	b_fc2 = bias([n_hidden_2])
	h_2   = tf.nn.softplus(tf.matmul(h_1, W_fc2) + b_fc2)

	W_fc3 = weights([n_hidden_2, n_hidden_3]) 
	b_fc3 = bias([n_hidden_3])
	h_3   = tf.nn.softplus(tf.matmul(h_2, W_fc3) + b_fc3)

	# Parameters for the Gaussian
	z_mu = tf.add(tf.matmul(h_3, weights([n_hidden_3, n_z])), bias([n_z]))
	# A little trick:
	#  sigma is always > 0.
	#  We don't want to enforce that the network produces only positive numbers, therefore we let 
	#  the network model the parameter log(\sigma^2) $\in [\infty, \infty]$
	z_ls2 = tf.add(tf.matmul(h_3, weights([n_hidden_3, n_z])), bias([n_z]))
	return z_mu, z_ls2
#### DECODER NETWORK ####
def decoder(z_ls2, z_mu, n_z=2, x_shape=784):
	eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32) # Adding a random number
	z = tf.add(z_mu, tf.sqrt(tf.exp(z_ls2))* eps)  # The sampled z

	n_hidden_1 = 500
	n_hidden_2 = 501
	n_hidden_3 = 503

	W_fc1_g = weights([n_z, n_hidden_1])
	b_fc1_g = bias([n_hidden_1])
	h_1_g   = tf.nn.softplus(tf.matmul(z, W_fc1_g) + b_fc1_g)

	W_fc2_g = weights([n_hidden_1, n_hidden_2])
	b_fc2_g = bias([n_hidden_2])
	h_2_g   = tf.nn.softplus(tf.matmul(h_1_g, W_fc2_g) + b_fc2_g)

	W_fc3_g = weights([n_hidden_2, n_hidden_3])
	b_fc3_g = bias([n_hidden_3])
	h_3_g   = tf.nn.softplus(tf.matmul(h_2_g, W_fc3_g) + b_fc3_g)

	x_mu = tf.add(tf.matmul(h_3_g,  weights([n_hidden_3, x_shape])), bias([x_shape]))
	x_ls2 = tf.add(tf.matmul(h_3_g,  weights([n_hidden_3, x_shape])), bias([x_shape]))
        x_reconstr_mean = tf.nn.sigmoid(x_mu)
        x_reconstr_mean = tf.clip_by_value(x_reconstr_mean, 1e-8, 1-1e-8)
        
	return x_mu, x_ls2, z, x_reconstr_mean

####### OPTIMIZER #######
def loss(x, x_ls2, x_mu, z_ls2, z_mu):
	#reconstr_loss = tf.reduce_sum(0.5 * x_ls2 + (tf.square(x-x_mu)/(2.0 * tf.exp(x_ls2))), 1)
        #x_reconstr_mean = tf.nn.sigmoid(x_mu)
        x_reconstr_mean = x_mu
	reconstr_loss = -tf.reduce_sum(x * tf.log(1e-5 + x_reconstr_mean) + (1-x) * tf.log(1e-5 + 1 - x_reconstr_mean), 1)
	latent_loss = -0.5 * tf.reduce_sum(1 + z_ls2 - tf.square(z_mu) - tf.exp(z_ls2), 1)
	cost_elem = reconstr_loss + latent_loss   # average over batch
        total_cost = tf.reduce_mean(cost_elem)
	return total_cost, latent_loss, reconstr_loss, cost_elem
	# Use ADAM optimizer

def vae(x, n_z=2):
	z_mu, z_ls2 = encoder(x, n_z)
	x_mu, x_ls2, z, x_reconstr_mean = decoder(z_mu, z_ls2, n_z)
	#cost, latent_loss, reconstr_loss, cost_elem = loss(x, x_ls2, x_mu, z_ls2, z_mu)
	cost, latent_loss, reconstr_loss, cost_elem = loss(x, x_ls2, x_reconstr_mean, z_ls2, z_mu)
	return {
                'cost_elem': cost_elem,
		'cost':cost,
		'latent_loss':latent_loss, 
		'reconstr_loss':reconstr_loss,
		'x_ls2':x_ls2,
		'x_mu':x_mu,
		'x_reconstr_mean':x_reconstr_mean,
		'z_ls2':z_ls2,
		'z_mu':z_mu}

if __name__ == '__main__':
	# data
	#xx = next_batch(1000)
	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#plt.plot(xx[:,0], xx[:,1], '.')
	#plt.show()
	#plt.show()
	#print xx.shape
	######## CREATE MODEL ###########
        os.system('mkdir model_mnist; mkdir mnist_figs')
	x = tf.placeholder("float32", shape=[None, 784]) #Batchsize x Number of Pixels
        n_z = 2
	z_mu, z_ls2 = encoder(x, n_z)
	x_mu, x_ls2, z, x_reconstr_mean = decoder(z_mu, z_ls2, n_z)
	cost, latent_loss, reconstr_loss, cost_elem= loss(x, x_ls2, x_reconstr_mean, z_ls2, z_mu)

	### Optimizer ###
	optimizer =  tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	runs = 40000#Set to 0, for no training
	init = tf.initialize_all_variables()
	saver = tf.train.Saver()

	#data = next_batch(batch_size)
        test_data, test_labels  = get_clean(num=500, labels=range(5) )
	test_batch = my_next_batch(1, test_data)
        data, data_labels  = get_clean(num=4000, labels=range(5) )
	batch = my_next_batch(batch_size, data)
	with tf.Session() as sess:
	    sess.run(init)
            load_model = False 
	    if load_model:
		saver.restore(sess, "model_mnist/model.ckpt")

	    #batch_xs = next_batch(batch_size)
            batch_xs = batch.next() 
	    print(batch_xs.shape)
	    dd = sess.run([cost], feed_dict={x: batch_xs})
	    print('Test run after starting {}'.format(dd))

	    for epoch in range(runs):
		avg_cost = 0.
		elfbatch_xs = next_batch(batch_size)
                batch_xs = batch.next()
		_,d, z_mean_val, z_log_sigma_sq_val, elem = sess.run((optimizer, cost, z_mu, z_ls2, cost_elem), feed_dict={x: batch_xs})
		avg_cost += d / batch_size

		# Display logs per epoch step
		if epoch % 100 == 0:
		    save_path = saver.save(sess, "model_mnist/model.ckpt") #Saves the weights (not the graph)
		    print ("Model saved in file: {}".format(save_path))
		    print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
		    print ("{} {} mean sigma2 {}".format(z_mean_val.min(), z_mean_val.max(), np.mean(np.exp(z_log_sigma_sq_val))))

		if epoch % 1000 == 0:
            ###--- Epoch Visualization: SPREAD###
		    N = 5
		    num_n = 4000
		    px_map = np.zeros((num_n*N, 3))
		    count = 0
		    test_batch = my_next_batch(batch_size, data)
		    test_labels = data_labels
		    for test_ep in range(num_n*N/batch_size):
			batch_xs = test_batch.next()
			elem, z_value = sess.run((cost_elem, z_mu), feed_dict={x: batch_xs})
			px_map[count*batch_size:(count+1)*batch_size,:2] = z_value
			px_map[count*batch_size:(count+1)*batch_size,2] = elem 
			count+=1
		    print 'MIN', px_map[:,2].min(), 'MAX', px_map[:,2].max()
		    plt.figure(figsize=(8, 10))        
		    plt.scatter(px_map[:,0], px_map[:,1], c=test_labels, edgecolor='none', cmap=discrete_cmap(N, 'jet')) 
		    plt.colorbar(ticks=range(N)) 
		    plt.savefig('mnist_figs/latent_space_color'+str(epoch)+'.png') 
        ##------- Visualization:  Generated Samples for Different Values of Z-------####
	    nx = ny = 20
	    x_values = np.linspace(-20, 13, nx)
	    y_values = np.linspace(-20, 13, ny)
            d = np.zeros([batch_size,n_z],dtype='float32')
	    canvas = np.empty((28*ny, 28*nx))
	    for i, yi in enumerate(x_values):
	    	for j, xi in enumerate(y_values):
	    	    z_mean = np.array([[xi, yi]])
	    	    d[0] = z_mean
                    #d[0, :] = np.random.normal(size=(1,8))
	    	    x_mean = sess.run(x_reconstr_mean, feed_dict={z: d})
	    	    canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

            plt.figure(figsize=(8, 10))        
            Xi, Yi = np.meshgrid(x_values, y_values)
            plt.imshow(canvas, origin="upper", vmin=0, vmax=1,interpolation='none',cmap=plt.get_cmap('gray'))
            plt.tight_layout()
            plt.savefig('mnist_figs/samples_of_z_mnist.png') 
        ##-------- Visualization-END --------###
        ##------- Visualization:  HEATMAP OF P(x) -------####
            #test_batch_xs = test_batch.next() 
            N = 5
            num_n = 2000
            px_map = np.zeros((num_n*N, 3))
            count = 0
	    test_batch = my_next_batch(batch_size, data)
            test_labels = data_labels
            for test_ep in range(num_n*N/batch_size):
                batch_xs = test_batch.next()
		elem, z_value = sess.run((cost_elem, z_mu), feed_dict={x: batch_xs})
                px_map[count*batch_size:(count+1)*batch_size,:2] = z_value
                px_map[count*batch_size:(count+1)*batch_size,2] = elem 
                count+=1
            plt.figure(figsize=(8, 10))        
	    plt.scatter(px_map[:,0], px_map[:,1], c=px_map[:,2], cmap='autumn') 
            plt.savefig('mnist_figs/latent_space_heatmap.png') 
            print 'MIN', px_map[:,2].min(), 'MAX', px_map[:,2].max()
            plt.colorbar()
            plt.figure(figsize=(8, 10))        
	    plt.scatter(px_map[:,0], px_map[:,1], c=test_labels, edgecolor='none', cmap=discrete_cmap(N, 'jet')) 
            plt.colorbar(ticks=range(N)) 
            plt.savefig('mnist_figs/latent_space_color.png') 
        ##-------- Visualization-END --------###
        ##-------- Visualization: The latent distribution P(z|x) of the training set --------###
	    all_z = np.zeros((1,8))
            n_samples = data.shape[0]
	    total_batch = int(n_samples / batch_size)
	    batch = my_next_batch(batch_size, data)
	    # Loop over all batches
	    for i in range(total_batch):
		batch_xs = batch.next() 
		x_reconstruct,z_vals,z_mean_val,z_log_sigma_sq_val  = \
		sess.run((x_reconstr_mean, z, z_mu, z_ls2), feed_dict={x: batch_xs})
		all_z = np.vstack((all_z, z_mean_val)) 
            plt.figure(figsize=(25,10))
	    plt.subplot(1,2,1)
	    plt.scatter(all_z[:,0], all_z[:,1])
	    plt.xlim(np.min(all_z[:,0]),np.max(all_z[:,0]))
	    plt.ylim(np.min(all_z[:,1]),np.max(all_z[:,1]))
	    #plt.ylim(-3,3)
	    plt.subplot(1,2,2)
	    plt.hist2d(all_z[:,0], all_z[:,1], (75, 75), cmap=plt.cm.jet)
	    plt.colorbar()
            plt.savefig('mnist_figs/latent_distribution.png')
        ##------ Visualisation-END ------####
	#m(-3,3)
5
