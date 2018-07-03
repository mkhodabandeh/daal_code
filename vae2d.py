import os, sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

batch_size = 50 #We have to define the batch size with the current version of TensorFlow

def to_one_hot(Y, n=None):
    if not n:
	    n = int(np.max(Y))+1
    y = np.zeros((Y.shape[0], n))
    goods = (Y<n)
    y[goods, Y[goods].astype(int)] = 1
    return y
 
def get_val(num=200, onehot=False):
    d1 = next_batch(num) 
    d2 = next_batch(num) 
    d2[:,1] *= -1
    d2[:,1] += 0.2 
    d2[:,0] = 0.55-d2[:,0]
    Y = np.ones((2*num,))
    Y[0:num] *= 0
    X = np.concatenate((d1,d2), axis=0)
    if onehot:
        Y = to_one_hot(Y)
    return X, Y

def get_noisy_data(num=5000, outlier_percent=0.6, onehot=False):
    ### Good data ###
    X, Y = get_val(num=num)
    ### Noisy data ##
    noisy_num = int(outlier_percent*num)
    print 'Number of Noisy Data:', noisy_num, 'Total number of data:', noisy_num+2*num
    x = np.random.uniform(2*np.min(X), 2*np.max(X), size=(noisy_num,2))
    y = np.ones((noisy_num,))*2
    ### Combined  ###
    X = np.concatenate((X, x), axis=0) 
    Y = np.concatenate((Y, y), axis=0) 
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
def encoder(x, n_z=1):

	#n_z = 1 #Dimension of the latent space
	# Input
	#x = tf.placeholder("float32", shape=[None, 2]) #Batchsize x Number of Pixels
	n_hidden_1 = 5
	n_hidden_2 = 6

	# First hidden layer
	W_fc1 = weights([2, n_hidden_1])
	b_fc1 = bias([n_hidden_1])
	h_1   = tf.nn.softplus(tf.matmul(x, W_fc1) + b_fc1)

	# Second hidden layer
	W_fc2 = weights([n_hidden_1, n_hidden_2]) 
	b_fc2 = bias([n_hidden_2])
	h_2   = tf.nn.softplus(tf.matmul(h_1, W_fc2) + b_fc2)


	# Parameters for the Gaussian
	z_mu = tf.add(tf.matmul(h_2, weights([n_hidden_2, n_z])), bias([n_z]))
	# A little trick:
	#  sigma is always > 0.
	#  We don't want to enforce that the network produces only positive numbers, therefore we let 
	#  the network model the parameter log(\sigma^2) $\in [\infty, \infty]$
	z_ls2 = tf.add(tf.matmul(h_2, weights([n_hidden_2, n_z])), bias([n_z]))
	return z_mu, z_ls2
#### DECODER NETWORK ####
def decoder(z_ls2, z_mu, n_z=1):
	eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32) # Adding a random number
	z = tf.add(z_mu, tf.sqrt(tf.exp(z_ls2))* eps)  # The sampled z

	n_hidden_1 = 5
	n_hidden_2 = 6

	W_fc1_g = weights([n_z, n_hidden_1])
	b_fc1_g = bias([n_hidden_1])
	h_1_g   = tf.nn.softplus(tf.matmul(z, W_fc1_g) + b_fc1_g)

	W_fc2_g = weights([n_hidden_1, n_hidden_2])
	b_fc2_g = bias([n_hidden_2])
	h_2_g   = tf.nn.softplus(tf.matmul(h_1_g, W_fc2_g) + b_fc2_g)

	x_mu = tf.add(tf.matmul(h_2_g,  weights([n_hidden_2, 2])), bias([2]))
	x_ls2 = tf.add(tf.matmul(h_2_g,  weights([n_hidden_2, 2])), bias([2]))
	return x_mu, x_ls2, z

####### OPTIMIZER #######
def loss(x, x_ls2, x_mu, z_ls2, z_mu):
	reconstr_loss = tf.reduce_sum(0.5 * x_ls2 + (tf.square(x-x_mu)/(2.0 * tf.exp(x_ls2))), 1)
	latent_loss = -0.5 * tf.reduce_sum(1 + z_ls2 - tf.square(z_mu) - tf.exp(z_ls2), 1)
	cost = reconstr_loss + latent_loss   # average over batch
	return tf.reduce_mean(cost), latent_loss, reconstr_loss, cost
	# Use ADAM optimizer

def vae(x, n_z=2):
	z_mu, z_ls2 = encoder(x, n_z)
	x_mu, x_ls2, z = decoder(z_mu, z_ls2, n_z)
	cost, latent_loss, reconstr_loss, cost_elem = loss(x, x_ls2, x_mu, z_ls2, z_mu)
	return {
                'cost_elem': cost_elem,
		'cost':cost,
		'latent_loss':latent_loss, 
		'reconstr_loss':reconstr_loss,
		'x_ls2':x_ls2,
		'x_mu':x_mu,
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
	x = tf.placeholder("float32", shape=[None, 2]) #Batchsize x Number of Pixels
        n_z = 2 
	z_mu, z_ls2 = encoder(x, n_z)
	x_mu, x_ls2, z = decoder(z_mu, z_ls2, n_z)
	cost, latent_loss, reconstr_loss, cost_elem= loss(x, x_ls2, x_mu, z_ls2, z_mu)
        dirpath = './vae2d_plots/'
        os.system('mkdir -p '+dirpath)
	### Optimizer ###
	optimizer =  tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

	runs = 60000 #Set to 0, for no training
	init = tf.initialize_all_variables()
	saver = tf.train.Saver()

	data = next_batch(2000)
	batch = my_next_batch(batch_size, data)
	with tf.Session() as sess:
	    sess.run(init)
	    #batch_xs = next_batch(batch_size)
            batch_xs = batch.next() 
	    print(batch_xs.shape)
	    dd = sess.run([cost], feed_dict={x: batch_xs})
	    print('Test run after starting {}'.format(dd))

	    for epoch in range(runs):
		avg_cost = 0.
		#batch_xs = next_batch(batch_size)
                batch_xs = batch.next()
		_,d, z_mean_val, z_log_sigma_sq_val, elem = sess.run((optimizer, cost, z_mu, z_ls2, cost_elem), feed_dict={x: batch_xs})
		avg_cost += d / batch_size

		# Display logs per epoch step
		if epoch % 50 == 0:
		    save_path = saver.save(sess, "model_2d/model2d.ckpt") #Saves the weights (not the graph)
		    print ("Model saved in file: {}".format(save_path))
		    print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
		    print ("{} {} mean sigma2 {}".format(z_mean_val.min(), z_mean_val.max(), np.mean(np.exp(z_log_sigma_sq_val))))



	saver = tf.train.Saver()
	with tf.Session() as sess:
	    saver.restore(sess, "model_2d/model2d.ckpt")
	    x_sample = next_batch(batch_size)
	    var = (x_mu, x_ls2, z, z_mu, z_ls2, cost, reconstr_loss, latent_loss)
	    #vv = tf.add(x_mu, tf.sqrt(tf.exp(x_ls2))* eps)  # The sampled z
	    out = sess.run(var, feed_dict={x: x_sample})
	    x_mu_val, x_ls2_val, z_vals, z_mu_val,z_ls2_val, cost_val, reconstr_loss_val,latent_loss_val = out

	    fig = plt.figure()
	    plt.plot(x_mu_val[:,0], x_mu_val[:,1], '.')
	    plt.plot(x_sample[:,0], x_sample[:,1], '+')
	    plt.savefig('res.png')


            z_vals1 = np.reshape(np.asarray(np.linspace(-8,8, batch_size), dtype='float32'), (batch_size,1))
            z_vals = np.ones((batch_size, 2)) * 0 
            z_vals[:,1] = z_vals1[:,0]
	    eps = 1*tf.random_normal((batch_size, 1), 0, 1, dtype=tf.float32) # Adding a random number
	    vv = tf.add(x_mu, tf.sqrt(tf.exp(x_ls2))* eps)  # The sampled z
            [vv_val, x_mu_val, x_ls2_val] = sess.run([vv,x_mu, x_ls2], feed_dict={z: z_vals})
            #plt.plot(vv_val[:,0], vv_val[:,1], '*')
            idx = np.linspace(0, batch_size-1, 20, dtype='int32')
            #print z_vals.shape
            #print idx.shape
	    fig = plt.figure()
            plt.scatter(x_mu_val[idx,0], x_mu_val[idx,1], c=np.array(range(20)), s=60000* np.mean(np.exp(x_ls2_val[idx,:]), axis=1))
            plt.savefig(dirpath+'mu_var.png')
	    #plt.show()

            #Plotting the heatmap of Px

	    x_minmax = (np.min(data[:,0]), np.max(data[:,0]))
	    y_minmax = (np.min(data[:,1]), np.max(data[:,1]))
	    grid = get_grid(y_minmax, y_minmax, 250)
	    train = get_noisy_data(1000, 2)[0]
            px_map = run_all(sess, reconstr_loss, x, grid, batch_size=batch_size)
            p_data_map = run_all(sess, reconstr_loss, x, train, batch_size=batch_size)
	    mpx_map= px_map.reshape((len(px_map), 1))
	    p_data_map= p_data_map.reshape((len(p_data_map), 1))
            for c in [0.01,0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0,1.4,2]:
		    dpx_map = np.exp(-c*mpx_map)
		    ipx_map = np.concatenate((grid, dpx_map), axis=1)
		    plt.figure()
		    plt.scatter(ipx_map[:,0], ipx_map[:,1], c=ipx_map[:,2], cmap='autumn_r', marker='o', s=6 ) 
                    plt.colorbar()
		    plt.savefig(dirpath+'px_map_{}.png'.format(str(c)))
		    dpx_map = np.exp(-c*p_data_map)
		    ipx_map = np.concatenate((train, dpx_map), axis=1)
		    plt.figure()
		    plt.scatter(ipx_map[:,0], ipx_map[:,1], c=ipx_map[:,2], cmap='autumn_r', marker='o', s=6 ) 
                    plt.colorbar()
		    plt.savefig(dirpath+'p_data_map_{}.png'.format(str(c)))
            print 'DONE'
