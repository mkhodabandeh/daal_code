import argparse
import time
import copy
import os,sys
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import vae
import json

from keras.models import Sequential
from keras.layers import *
import tensorflow as tf
from keras.datasets import cifar10

#################### DATA ##################
def get_mnist():
    train, test = cifar10.load_data(label_mode='fine')
    return train, test

def to_one_hot(Y, n=None):
    if not n:
	    n = int(np.max(Y))+1
    y = np.zeros((Y.shape[0], n))
    goods = (Y<n)
    y[goods, Y[goods].astype(int)] = 1
    return y

def get_val(num=1000, onehot=True, labels=range(5)):
    (train_x, train_y), (test_x, test_y) = get_mnist()
    X = np.zeros((len(labels)*num, test_x.shape[1]))
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
 
def next_batch(batch_size, non_crossing=True):
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

################## CLASSIFIER ###################
def get_classifier1(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def get_classifier(input_dim, num_classes):
    model = Sequential()
    model.add(Reshape((-1, 32,32), input_shape=(input_dim,)))
    #model.add(Convolution2D(20, 5, 5, border_mode="same",
    #        input_shape=(depth, height, width)))
    model.add(Conv2D(20, kernel_size=(5, 5), activation='relu', border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, kernel_size=(5, 5), activation='relu', border_mode="same"))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))    
    return model

def evaluate(train,test,U, model, epochs=100, batch_size=10):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X,Y = train
    X_t, Y_t = test
    #print 'Fitting the model ...'
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)
    #print 'Evaluating ...'
    accuracy = model.evaluate(X_t, Y_t, verbose=0)
    #print 'Predicting ...'
    predictions = model.predict(U)
    return accuracy[1], predictions # returns 1- accuracy on validation set 2- predictions on unlabeled set

def get_boundary(model, grid):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    predictions = model.predict(grid)
    indices = np.abs(predictions[:,0] - 0.5) < 0.05
    return grid[indices]


############### UTILS ##########################
def half_round(n):
    return int(n*2)/2.0

def get_label_histogram(labels):
    class_count = [0]*10
    for yyy in labels:
      class_count[int(yyy)]+=1
    return class_count

def save_scattered_images(z, id, path='scattered_image.jpg', z_range=None, title=None, gray=None):
    N = 10
    if z_range is None:
       z_min = np.min(z, 0)
       z_max = np.max(z, 0)
       z_range = (z_min, z_max)
    else:
       z_min, z_max = z_range[0], z_range[1]
    plt.figure(figsize=(8, 6))
    plt.axis([z_min[0], z_max[0], z_min[1], z_max[1]])
    if gray is not None:
       plt.scatter(gray[:, 0], gray[:, 1], c='lightgrey', marker='o', alpha=0.16, edgecolor='none') 
    if len(id.shape) == 1:
       plt.scatter(z[:, 0], z[:, 1], c=id, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    else: 
       plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    if title is not None:
       plt.title(title)
    #print 'MIN', z_min, 'MAX',z_max
    plt.colorbar(ticks=range(N))
    #axes = plt.gca()
    #axes.set_xlim([1.5*z_min[0], 1.5*z_max[0]])
    #axes.set_ylim([1.5*z_min[1], 1.5*z_max[1]])
    plt.grid(True)
    plt.savefig(path)
    plt.close('all')
    return z_range
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_entropy(preds):
    return np.sum(-1*preds*np.log(preds+np.finfo(np.float32).eps), axis=1) 

def normalize(arr):
    arr = arr.copy() - np.min(arr)+1e-20
    arr = arr / np.max(arr)
    return arr

def copy_shuffle(X,Y,Px_val, zs=None):
    newX,newY = X.copy(), Y.copy() #TODO better not to copy and play with indices, instead. whatever!
    newPx = Px_val[:]
    #assert len(scores) == train_data[0].shape[0]
    indices = range(len(Px_val))
    random.shuffle(indices)
    #rng_state = np.random.get_state()
    #np.random.shuffle(newX)
    #np.random.set_state(rng_state)
    #np.random.shuffle(newY)
    #np.random.set_state(rng_state)
    #np.random.shuffle(newPx)
    newX = newX[indices]
    newY = newY[indices]
    newPx = newPx[indices]
    if zs is not None:
       newZs = zs.copy()
       #np.random.set_state(rng_state)
       #np.random.shuffle(newZs)
       newZs = newZs[indices]
       return newX,newY,newPx,newZs 
    return newX,newY,newPx

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

##############################################
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

############## VISUALIZATIONS ################
 
def visualize(X, scores, candids=None, noise=None, px_map=None, filename=None, boundary=None, isShow=False,labels=None):
    plt.close('all')
    if px_map is not None: 
        fig=plt.figure()
	plt.scatter(px_map[:,0], px_map[:,1], c=px_map[:,2], cmap='autumn') 
        f,fmt = filename.split('.')
        plt.savefig(f+'_map.'+fmt)
        plt.close(fig)
    fig=plt.figure()
    if type(scores) in [list, tuple]:
        start = (0 if labels is None else 1) 
	fig, axis = plt.subplots(nrows=1, ncols=3+start, figsize=(20, 6), sharey=True)       
        if labels is not None:
            axis[0].scatter(X[:,0], X[:,1], c=labels, marker='.') 
            axis[0].set(adjustable='box-forced', aspect='equal')
	if boundary is not None:
            axis[start].plot(boundary[:,0], boundary[:,1], 'k^', markersize=1)
	for i in range(start,len(scores)+start):
	  axis[i].scatter(X[:,0], X[:,1], c=scores[i-start], marker='.', cmap='autumn')
          axis[i].set(adjustable='box-forced', aspect='equal')
        ax = axis[-1]
    else:
        ax = plt
	ax.scatter(X[:,0], X[:,1], c=scores,  marker='.', cmap='autumn')
    if candids is not None:
	    ax.scatter(candids[:,0], candids[:,1], marker='o', color='#00ff00', s=10)
    if noise is not None:
	    ax.scatter(noise[:,0], noise[:,1], marker='+', color='blue', s=10)
    if filename is None:
       filename = 'fig.png'
    plt.savefig(filename)
    if isShow:
       show()
    else:
       plt.close(fig)

##---------------------
def do_daal(predictions, Px, BETA):
    entropies = get_entropy(predictions)#lower the better
    #entropies = normalize(entropies)
    #entropies -> less better [0-1]
    # higher entropy --> more uncertain
    # later we will choose the higher ones 
    # if Px is low that means x is probably an outlier
    # So I multiply Entropy by (1-Px) so outliers will have a larger valie  
    #Assumption: 0<=Px<=1 && 0<=Entropy<=1
    #weighted_entropies = entropies*np.power(1-Px, BETA)     	    # we would like to select the largest
    new_Px = np.power(Px, BETA)
    #new_Px = np.max(new_Px)-new_Px
    weighted_entropies = entropies*(new_Px)     	    # we would like to select the largest
    argsorted = np.argsort(weighted_entropies)[::-1]#increasing sort, then reversed 
    return argsorted    

##-----------------------
def plot_samples(classifier, train_sample_, Px, BETA, save_to='top_bottom.png'):
    predictions = classifier.predict(train_sample_)
    train_sample = train_sample_.reshape((-1, 32, 32))
    argsorted_beta = do_daal(predictions, Px, BETA)
    argsorted_0 = do_daal(predictions, Px, 0)
    #print 'argsorted_bata.shape', argsorted_beta.shape
    #print 'argsorted_bata', argsorted_beta
    #print 'train_sample.shape', train_sample.shape
    top = np.concatenate(train_sample[argsorted_beta], axis=1)
    bottom = np.concatenate(train_sample[argsorted_0], axis=1)
    canvas = np.concatenate((top,bottom), axis=0)
    plt.figure()
    plt.imshow(canvas, origin="upper", vmin=0, vmax=1,interpolation='none',cmap=plt.get_cmap('gray'))
    plt.tight_layout()
    plt.savefig(save_to, dpi=1000) 

##-----------------------
def plot_candidates(candidates, X, save_path):
    canvas = np.concatenate(X[candidates].reshape((-1,32,32)), axis=1)
    plt.figure()
    plt.imshow(canvas, origin="upper", vmin=0, vmax=1,interpolation='none',cmap=plt.get_cmap('gray'))
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000) 

#######################################################
if __name__=="__main__":
    desc = "Distribution Aware Active Learning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--beta', type=str, default='0.8', help='Beta')
    parser.add_argument('--am', type=float, default=0.8, help='Anneal Multiplier')
    parser.add_argument('--outlier', type=float, default=2, help='Outlier to inlier portion')
    parser.add_argument('--init_size', type=int, default=20, help='Size of the initial training set')
    parser.add_argument('--dim_z', type=int, default=2, help='latent vae z', required=True)
    parser.add_argument('--step_size', type=int, default=50, help='Number of queries per active learning iteration')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of active learning iterations')
    parser.add_argument('--runs', type=int, default=1, help='Number of times to run the experiment')
    parser.add_argument('--exp_name' ,type=str, default='exp', help='name of the experiment folder', required=True)
    parser.add_argument('--init_type' , default='balanced', choices=['balanced','biased','beta'],help='how to select the initial labeled set')
    args = parser.parse_args()
    BETA = args.beta
    #iteration = int(sys.argv[2])
    exp_name = args.exp_name 
    os.system('mkdir '+exp_name)
    if ',' in BETA:
        BETAS = map(lambda x: float(x.strip()), BETA.split(','))
    else:
        BETAS = [float(BETA)]
    #run_id = sys.argv[2]
    flog = open(exp_name+'_log.txt', 'w')
    flog.write(str(vars(args))+'\n\n')
    #exp_dir = exp_name+'/id={0}_BETA={1}/'.format(run_id, str(BETA))
    #os.system('mkdir '+exp_dir+' -p')
    ############# CONSTANTS ####################
    INIT_TYPE = args.init_type
    INITIAL_TRAINING_SIZE = args.init_size
    OUTLIER_INLIER_PORTION = args.outlier
    ANNEAL_MULTIPLIER = args.am
    ACTIVE_LEARNING_ITERS = args.runs #instead I'll run on GPU cluster, in parallel (100)
    ACTIVE_LEARNING_STEP_SIZE = args.step_size 
    ACTIVE_LEARNING_STEPS = args.num_steps 
    MAX_EPOCH = 20#  Classification training maximum number of iterations
    CLASSIFIER_BS = 32
    if OUTLIER_INLIER_PORTION == 0:
       CLASS_NUM = 10
    else:
       CLASS_NUM = 5
    GRID_NUM = 250
    print '****** BETA IS : ', BETA, '********' 
    PLOT = False 
    ############ LOAD DATA ###############
    #clean_data = get_val(5000, True)
    val = get_val(892, onehot=True, labels=range(CLASS_NUM))
    if OUTLIER_INLIER_PORTION == 0:
       num= 1000 
       train = get_clean(num, onehot=False, labels=range(CLASS_NUM))
    else:
       num= 1000 
       train = get_noisy_data(num, outlier_percent=OUTLIER_INLIER_PORTION, onehot=False)
    #train_sample = get_noisy_data(10, outlier_percent=2, onehot=False)
    #if PLOT:
    #        num=500
    #        plt.scatter(val[0][:num,0], val[0][:num,1], facecolors='none', edgecolors='#93DB70', s=16, )
    #        plt.scatter(val[0][num:2*num,0], val[0][num:2*num,1], color='#87CEFA', s=16, marker='x')
    #        plt.savefig('data.pdf')
    #if PLOT:
    #        plt.figure()
    #        num=1000
    #        plt.scatter(train[0][:num,0], train[0][:num,1], facecolors='none', edgecolors='#93DB70', s=16, )
    #        plt.scatter(train[0][num:2*num,0], train[0][num:2*num,1], color='#87CEFA', s=16, marker='x')
    #        plt.scatter(train[0][2*num:,0], train[0][2*num:,1], color='#D3D3D3', s=6, marker='.')#, alpha=0.5)
    #        plt.savefig('data_pool.pdf')
    LEN = train[0].shape[0] 

    #x_minmax = (np.min(train[0][:,0]), np.max(train[0][:,0]))
    #y_minmax = (np.min(train[0][:,1]), np.max(train[0][:,1]))
    #x_minmax = np.array((-1.0, 1.0))
    #y_minmax = np.array((-1.0, 1.0))
    #print GRID_NUM
    #grid = get_grid(y_minmax, y_minmax, GRID_NUM)
    ############### LOAD VAE MODEL ####################
    # x   -> input
    # Px -> score
    """ build graph """
    dim_z = args.dim_z
     
    dim_img = 32**2
    n_hidden = 500
    learn_rate =  1e-3
    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for PMLR
    z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

    # network architecture
    y, z, loss, neg_marginal_likelihood, KL_divergence, px_elem  = vae.autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob)

    # optimization
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    # latent space for PMLR
    decoded = vae.decoder(z_in, dim_img, n_hidden)
    saver = tf.train.Saver()
    ########## Generative Model Scores on Unlabeled/Training set ########### 
    ## It needs to be done only "once" 
    batch_size = 50 


    import ipdb
    model_path = "models/model_class_"+str(CLASS_NUM)+"_dim_"+str(dim_z)+"_cifar.ckpt"
    print model_path
    with tf.Session() as session:
        session.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})
        
        saver.restore(session, model_path)
	z_VALS_ORIG, Px_val = session.run([z, px_elem], feed_dict={x_hat: train[0], x:train[0], keep_prob : 1})
	save_scattered_images(z_VALS_ORIG, train[1], path=exp_name+"/PMLR_map.png")
        #Px_val = run_all(session, cost, x_inp, train[0], batch_size=batch_size)
        #Px_val = session.run(cost, feed_dict={x: train[0]}) 

        #Px_val = np.exp(Px_val)
        #Px_val = sigmoid(Px_val)
        print '**** Min(Px)={},  Max(Px)={} ****'.format(np.min(Px_val), np.max(Px_val))
        Px_val = Px_val.reshape((LEN,))
        ipdb.set_trace()
        # For a subset of training set
        #samples_Px_val = run_all(session, cost, x_inp, train_sample[0], batch_size=batch_size)
        #samples_Px_val = np.exp(-samples_Px_val)
        #samples_Px = samples_Px_val.reshape((train_sample[0].shape[0],))
        #px_map = run_all(session, cost, x_inp, grid, batch_size=640)
        #px_map = run_all(session, cost, x_inp, train[0], batch_size=batch_size)
	#px_map= px_map.reshape((len(px_map), 1))
        #print px_map.shape, grid.shape
	#px_map = np.concatenate((train[0], px_map), axis=1)
	#if PLOT:
	#	plt.figure()
	#	px_map[:,2]= np.power(px_map[:,2], BETA)
	#	plt.scatter(px_map[:,0], px_map[:,1], c=px_map[:,2], cmap='autumn', marker='o', s=6 ) 
	#	plt.savefig('my_px_map.png')
########### Create a classifier #####################
	print Px_val
    print np.max(Px_val), np.min(Px_val)
    print train[0].shape[1], CLASS_NUM
    classifier= get_classifier(input_dim=train[0].shape[1], num_classes=CLASS_NUM)
    print 'CLASSIFIER', classifier
    weights = copy.deepcopy(classifier.get_weights())
    
    #do the experiment many times with different random_seeds
    daal_all= []
    all_candidates= []
    all_daal_size2acc = []
    for iteration in range(ACTIVE_LEARNING_ITERS):
        print "Active Learning Iteration:", iteration
        daal_accuracies = {str(BETA):[] for BETA in BETAS}
        daal_candidates = {str(BETA):[] for BETA in BETAS}
        daal_size2acc = {str(BETA):{} for BETA in BETAS}
        daal_noises= {str(BETA):[] for BETA in BETAS}
        print 'daal_accuracies: ', daal_accuracies
        #shuffle the training data
	X,Y,Px, z_VALS= copy_shuffle(train[0], train[1], Px_val, z_VALS_ORIG)
        Px = normalize(Px)
        #sample the initial training set
        if OUTLIER_INLIER_PORTION != 0:
           goods = np.nonzero(Y<CLASS_NUM)[0] # the reason that I'm doing this is because I don't want the initial training set to have outliers
        else:
           goods = range(LEN)
        goods_mask = np.array([False]*LEN)
        goods_mask[goods] = True
        labeled_ = np.array([False]*LEN)
        initial_labeled = []
        #initial_labeled =  random.sample(goods, INITIAL_TRAINING_SIZE)
        if INIT_TYPE == 'balanced':
	   per_class_num = INITIAL_TRAINING_SIZE/CLASS_NUM
	   for ll in range(CLASS_NUM):
	      goods_ll = np.nonzero(Y==ll)[0]
	      #initial_labeled.append(random.sample(goods_ll, INITIAL_TRAINING_SIZE/CLASS_NUM))
	      if per_class_num == 1:
	         initial_labeled.append(goods_ll[0])
	      else:
	         initial_labeled.extend(goods_ll[0:per_class_num])
        elif INIT_TYPE == 'biased':
           exclude = random.sample(range(CLASS_NUM), 5)
           init_pool = []
           for digit in exclude:
              init_pool.extend(np.nonzero(Y==digit)[0])
           random.shuffle(init_pool)
           initial_labeled = random.sample(init_pool, INITIAL_TRAINING_SIZE)
        elif INIT_TYPE == 'beta':
           if len(BETAS) != 1:
               raise Exception('if init_type==beta then BETA should be only one value not a list')
           temp_px = np.power(Px, BETAS[0])
           sorted_temp_px= np.argsort(temp_px)[::-1]#increasing sort, then reversed 
           print 'CHECK THIS OUT',set(Y[sorted_temp_px][:32])
           print 'CHECK THIS OUT',Y[sorted_temp_px][:32]
           initial_labeled = sorted_temp_px[0:INITIAL_TRAINING_SIZE]
        else:
           raise Exception('INIT_TYPE argument is wrong')
           #COMEBACK 
        print '###---- This is Initial Set histogram :', get_label_histogram(Y[initial_labeled])
        print '###---- This is Initial SET:', initial_labeled
        #np.save('initial_labeled_{}.npy'.format(iteration), initial_labeled)
        #initial_labeled = np.load('initial_labeled_{}.npy'.format(iteration))
        labeled_[initial_labeled] = True
        #train the classifier with the initial set
        all_labeled_= {str(BETA):labeled_.copy() for BETA in BETAS}
        
        train_l = (X[labeled_], to_one_hot(Y[labeled_], CLASS_NUM))
        #X_u = X[~labeled_] #unlabeled set
        classifier.set_weights(copy.deepcopy(weights))
	MAX_ITERS = MAX_EPOCH*sum(labeled_==True)/CLASSIFIER_BS

        acc, predictions = evaluate(train_l, val, X, classifier, MAX_ITERS, batch_size=CLASSIFIER_BS)	
        rounded_acc = half_round(acc*100)
        if rounded_acc not in daal_size2acc:
           daal_size2acc[rounded_acc] = []
	daal_size2acc[rounded_acc].append(INITIAL_TRAINING_SIZE)
         
	print "Accuracy:", acc
        #### FOR TESTING PURPOSES #####
        entropies = get_entropy(predictions)#lower the better
        entropies = normalize(entropies)
        exp_dir = exp_name+'/run={0}/'.format(iteration)
        os.system('rm -rf '+exp_dir)
        os.system('mkdir '+exp_dir+' -p')
        if dim_z == 2:
            z_range = save_scattered_images(z_VALS, Y, path=exp_dir+'/all_latent.png')
        np.save(exp_dir+'X.npy', X)
        np.save(exp_dir+'Y.npy', Y)
	np.save(exp_dir+'Px.npy', Px)
	np.save(exp_dir+'ent.npy', entropies)
        np.save(exp_dir+'initial.npy', initial_labeled)
        first_acc = acc
        ##### DONE ###
        #arr_range = np.arange(train_data[0].shape[0])
        for BETAA in BETAS:
           #exp_dir = exp_name+'/run={0}/BETA={1}/'.format(iteration,BETA)
           #os.system('mkdir {} -p'.format(exp_dir))
           daal_accuracies[str(BETAA)].append(first_acc)
           outliers = []
           labeled_ = all_labeled_[str(BETAA)]
           print len(np.nonzero(labeled_))
           all_noise = np.array([False]*LEN) 
           #plt.figure()
           for step in range(ACTIVE_LEARNING_STEPS):
               start_time = time.time()
               print ' '
               BETA = BETAA * (ANNEAL_MULTIPLIER**step)
               print "Step", step, 'Beta', BETA
               #get predicted probabilities of the unlabel set and compute the entropies
               #predictions = classifier_daal.predict(images) 
               entropies = get_entropy(predictions)#lower the better
    	       #entropies = normalize(entropies)
               #entropies -> less better [0-1]
               # higher entropy --> more uncertain
	       # later we will choose the higher ones 
               # if Px is low that means x is probably an outlier
               # So I multiply Entropy by (1-Px) so outliers will have a larger valie  
               #Assumption: 0<=Px<=1 && 0<=Entropy<=1
               #weighted_entropies = entropies*np.power(1-Px, BETA)     	    # we would like to select the largest
               new_Px = np.power(Px, BETA)
               #new_Px = np.max(new_Px)-new_Px
               weighted_entropies = entropies*(new_Px)     	    # we would like to select the largest
               argsorted = np.argsort(weighted_entropies)[::-1]#increasing sort, then reversed 
               #argsorted = np.argsort(weighted_entropies)#[::-1]#increasing sort, then reversed 
               #argsorted = arr_range[daal_set_indices][argsorted]
               candidates = []
               i = 0
               while len(candidates) < ACTIVE_LEARNING_STEP_SIZE:
                   if not labeled_[argsorted[i]]: 
                        candidates.append(argsorted[i])
                   i+=1
               daal_candidates[str(BETAA)].append(candidates)
               labeled_[candidates] = True
               noise = np.logical_and(labeled_, ~goods_mask)
               outliers.append(np.nonzero(noise)[0])
               labeled_ = np.logical_and(labeled_, goods_mask) # we filter out the outliers in this step
               daal_noises[str(BETAA)].append(np.sum(noise))
               print 'Number of OUTLIER queries', np.sum(noise)
               print 'Total Number of Labeled Data', np.sum(labeled_)
               label_histogram = get_label_histogram(Y[candidates])
               print 'Label Histograms', label_histogram 
               #++++------- VISUALIZATION -----++++# 
               #print train_sample[0].shape
               #plot_samples(classifier, train_sample[0], samples_Px, BETA, save_to=exp_dir+'train_orders_beta={0}_{1}.png'.format(BETA,step))
               #exp_dir = exp_name+'/run={0}/'.format(iteration)
               print 'labels:', Y[candidates]
               plot_candidates(candidates, X, save_path=exp_dir+'candidates_beta={0}_{1}.png'.format(BETA, step))
               #all_noise = np.logical_or(noise, all_noise)
               #boundary = get_boundary(classifier,grid)
               #print 'BOUNDARY', boundary.shape
               #visualize(X=X, 
	       #    	scores=(entropies, new_Px, weighted_entropies),
	       #    	#candids=X[candidates], 
	       #    	candids=X[candidates], 
               #            noise=X[noise], 
	       #    	filename='step_'+str(step)+'.png',
	       #    	boundary=boundary,
	       #    	labels=Y)
               #visualize(X=X, 
	       #    	scores=weighted_entropies, 
	       #    	#candids=X[candidates], 
	       #    	candids=X[labeled_], 
               #            noise=X[all_noise], 
	       #    	filename='all_until_step_'+str(step)+'.png',
	       #    	boundary=None,
	       #    	labels=Y)
               ##++++------- VISUALIZATION -----++++# 
               ##++++--- preparing for the next step: Train and Test with new data ---++++###
               train_l = (X[labeled_], to_one_hot(Y[labeled_], CLASS_NUM))
               classifier.set_weights(copy.deepcopy(weights))
              
               if dim_z == 2:
                  save_scattered_images(z_VALS[candidates], predictions[candidates], path=exp_dir+'/candidates_latent_{0}_before.png'.format(step), z_range=z_range, title='Before iteration '+str(step), gray=z_VALS)
                  save_scattered_images(z_VALS, predictions, path=exp_dir+'/all_latent_{1}_{0}_before.png'.format(BETA, step),z_range=z_range , title='Before iteration '+str(step))
               acc, predictions = evaluate(train_l, val, X, classifier, MAX_ITERS, batch_size=CLASSIFIER_BS)
               ###--- update daal_size2acc ----###
               rounded_acc = half_round(acc*100)
               if rounded_acc not in daal_size2acc:
                  daal_size2acc[rounded_acc] = []
               daal_size2acc[rounded_acc].append(step*ACTIVE_LEARNING_STEP_SIZE)
               ###--- end of updateing daal_size2acc ---###
               if dim_z == 2:
                  save_scattered_images(z_VALS, predictions, path=exp_dir+'/all_latent_{0}_after.png'.format(step),z_range=z_range, title='After iteration '+str(step))
                  save_scattered_images(z_VALS[candidates], predictions[candidates], path=exp_dir+'/candidates_latent_{1}_{0}_after.png'.format(BETA, step),z_range=z_range, title='After iteration '+str(step), gray=z_VALS)
               daal_accuracies[str(BETAA)].append(acc)
               print "Daal accuracy BETA={2} [step={0}]: {1}".format(step, acc, BETA) 
               print "---> Time:", time.time()-start_time
           flog.write('step: {0}'.format(step))
           print('Daal_accuracies: {}'.format(" ".join(map(lambda x: '{0:.02f}'.format(x*100),daal_accuracies[str(BETAA)]))))
           flog.write('Daal_accuracies: {0}'.format(" ".join(map(str,daal_accuracies[str(BETAA)]))))
           flog.flush()
        exp_dir = exp_name+'/run={0}/'.format(iteration)
        with open(exp_dir+'size2acc.json', 'w') as f:
           json.dump(daal_size2acc, f)
        with open(exp_dir+'accuracies.json', 'w') as f:
           json.dump(daal_accuracies, f)
        with open(exp_dir+'candidates.json', 'w') as f:
           json.dump(daal_candidates, f)
        with open(exp_dir+'num_outliers.json', 'w') as f:
           json.dump(daal_noises, f)

        ###--- PLOT accuracy ---###
        plt.figure()
        handles=[]
        for BETA in BETAS:
           h,=plt.plot(range(len(daal_accuracies[str(BETA)])),daal_accuracies[str(BETA)], 'x-', label='Beta={0:.02f}'.format(BETA))    
           handles.append(h)
        plt.legend(handles=handles)
        plt.savefig(exp_name+'/run={}/al_accs.png'.format(iteration))
        ###--- End of Plot---###
        daal_all.append(daal_accuracies)
        all_candidates.append(daal_candidates)
        all_daal_size2acc.append(daal_size2acc)
    with open(exp_name+'/all_size2acc.json', 'w') as f:
       json.dump(all_daal_size2acc, f)
    with open(exp_name+'/all_accuracies.json', 'w') as f:
       json.dump(daal_all, f)
    with open(exp_name+'/all_candidates.json', 'w') as f:
       json.dump(all_candidates, f)
    
    flog.close()
