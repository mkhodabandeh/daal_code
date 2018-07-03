import copy
import os,sys
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json

from keras.models import Sequential
from keras.layers import *
from vae_mnist import vae
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#################### DATA ##################
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
def get_classifier(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def get_classifier1(input_dim, num_classes):
    model = Sequential()
    model.add(Reshape((-1, 28,28), input_shape=(input_dim,)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
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
def get_entropy(preds):
    return np.sum(-1*preds*np.log(preds+np.finfo(np.float32).eps), axis=1) 

def normalize(arr):
    arr = arr.copy() - np.min(arr)
    arr = arr / np.max(arr)
    return arr

def copy_shuffle(X,Y,Px_val):
    newX,newY = X.copy(), Y.copy() #TODO better not to copy and play with indices, instead. whatever!
    Px = Px_val[:]
    #assert len(scores) == train_data[0].shape[0]
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)
    np.random.set_state(rng_state)
    np.random.shuffle(Px)
    return X,Y,Px

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
    train_sample = train_sample_.reshape((-1, 28, 28))
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
    canvas = np.concatenate(X[candidates].reshape((-1,28,28)), axis=1)
    plt.figure()
    plt.imshow(canvas, origin="upper", vmin=0, vmax=1,interpolation='none',cmap=plt.get_cmap('gray'))
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000) 

#######################################################
if __name__=="__main__":
    BETA = sys.argv[1]
    #iteration = int(sys.argv[2])
    exp_name = 'mnist_daal'
    if ',' in BETA:
        BETAS = map(lambda x: float(x.strip()), BETA.split(','))
    else:
        BETAS = [float(BETA)]
    #run_id = sys.argv[2]
    flog = open(exp_name+'_log.txt', 'w')
    #exp_dir = exp_name+'/id={0}_BETA={1}/'.format(run_id, str(BETA))
    #os.system('mkdir '+exp_dir+' -p')
    ############# CONSTANTS ####################
    INITIAL_TRAINING_SIZE = 5 
    ACTIVE_LEARNING_ITERS = 100 #instead I'll run on GPU cluster, in parallel (100)
    ACTIVE_LEARNING_STEP_SIZE = 50 
    ACTIVE_LEARNING_STEPS = 50 
    MAX_EPOCH=20#  Classification training maximum number of iterations
    CLASSIFIER_BS=10
    CLASS_NUM = 5 
    GRID_NUM = 250
    print '****** BETA IS : ', BETA, '********' 
    PLOT = False 
    ############ LOAD DATA ###############
    #clean_data = get_val(5000, True)
    num=800
    val = get_val(900, onehot=True, labels=range(CLASS_NUM))
    train = get_noisy_data(num, outlier_percent=4, onehot=False)
    train_sample = get_noisy_data(10, outlier_percent=2, onehot=False)
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
    x_inp = tf.placeholder('float32', shape=[None, train[0].shape[1]]) 
    print train[0].shape
    print x_inp
    vae_model = vae(x_inp, n_z=2)
    #cost = vae_model['latent_loss']
    cost = vae_model['reconstr_loss']
    #cost = vae_model['cost_elem']
    saver = tf.train.Saver()
    ########## Generative Model Scores on Unlabeled/Training set ########### 
    ## It needs to be done only "once" 
    batch_size = 50 
    with tf.Session() as session:
        saver.restore(session, 'model_mnist/model.ckpt')
        Px_val = run_all(session, cost, x_inp, train[0], batch_size=batch_size)
        #Px_val = session.run(cost, feed_dict={x: train[0]}) 
        Px_val = np.exp(-Px_val)
        print '**** Min(Px)={},  Max(Px)={} ****'.format(np.min(Px_val), np.max(Px_val))
        Px_val = Px_val.reshape((LEN,))
        # For a subset of training set
        samples_Px_val = run_all(session, cost, x_inp, train_sample[0], batch_size=batch_size)
        samples_Px_val = np.exp(-samples_Px_val)
        samples_Px = samples_Px_val.reshape((train_sample[0].shape[0],))
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
    weights = copy.deepcopy(classifier.get_weights())
    
    #do the experiment many times with different random_seeds
    daal_all= []
    all_candidates= []
    for iteration in range(ACTIVE_LEARNING_ITERS):
        print "Active Learning Iteration:", iteration
        daal_accuracies = {str(BETA):[] for BETA in BETAS}
        daal_candidates = {str(BETA):[] for BETA in BETAS}
        daal_noises= {str(BETA):[] for BETA in BETAS}
        print 'daal_accuracies: ', daal_accuracies
        #shuffle the training data
        X,Y,Px = copy_shuffle(train[0], train[1], Px_val)
        Px = normalize(Px)
        #sample the initial training set
        goods = np.nonzero(Y<5)[0] # the reason that I'm doing this is because I don't want the initial training set to have outliers
        goods_mask = np.array([False]*LEN)
        goods_mask[goods] = True
        labeled_ = np.array([False]*LEN)
        initial_labeled = []
        #initial_labeled =  random.sample(goods, INITIAL_TRAINING_SIZE)
        per_class_num = INITIAL_TRAINING_SIZE/CLASS_NUM
        for ll in range(CLASS_NUM):
           goods_ll = np.nonzero(Y==ll)[0]
           #initial_labeled.append(random.sample(goods_ll, INITIAL_TRAINING_SIZE/CLASS_NUM))
           if per_class_num == 1:
		   initial_labeled.append(goods_ll[0])
           else:
		   initial_labeled.extend(goods_ll[0:per_class_num])

        #np.save('initial_labeled_{}.npy'.format(iteration), initial_labeled)
        #initial_labeled = np.load('initial_labeled_{}.npy'.format(iteration))
        labeled_[initial_labeled] = True
        #train the classifier with the initial set
        all_labeled_= {str(BETA):labeled_.copy() for BETA in BETAS}
        
        train_l = (X[labeled_], to_one_hot(Y[labeled_]))
        #X_u = X[~labeled_] #unlabeled set
        classifier.set_weights(copy.deepcopy(weights))
	MAX_ITERS = MAX_EPOCH*sum(labeled_==True)/CLASSIFIER_BS

        acc, predictions = evaluate(train_l, val, X, classifier, MAX_ITERS, batch_size=CLASSIFIER_BS)	
	print "Accuracy:", acc
        #### FOR TESTING PURPOSES #####
        entropies = get_entropy(predictions)#lower the better
        entropies = normalize(entropies)
        exp_dir = exp_name+'/run={0}/'.format(iteration)
        os.system('mkdir '+exp_dir+' -p')
        np.save(exp_dir+'X.npy', X)
        np.save(exp_dir+'Y.npy', Y)
	np.save(exp_dir+'Px.npy', Px)
	np.save(exp_dir+'ent.npy', entropies)
        np.save(exp_dir+'initial.npy', initial_labeled)
        first_acc = acc
        ##### DONE ###
        #arr_range = np.arange(train_data[0].shape[0])
        for BETA in BETAS:
           #exp_dir = exp_name+'/run={0}/BETA={1}/'.format(iteration,BETA)
           #os.system('mkdir {} -p'.format(exp_dir))
           daal_accuracies[str(BETA)].append(first_acc)
           outliers = []
           labeled_ = all_labeled_[str(BETA)]
           print len(np.nonzero(labeled_))
           all_noise = np.array([False]*LEN) 
           #plt.figure()
           for step in range(ACTIVE_LEARNING_STEPS):
               print ' '
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
               #argsorted = arr_range[daal_set_indices][argsorted]
               candidates = []
               i = 0
               while len(candidates) < ACTIVE_LEARNING_STEP_SIZE:
                   if not labeled_[argsorted[i]]: 
                        candidates.append(argsorted[i])
                   i+=1
               daal_candidates[str(BETA)].append(candidates)
               labeled_[candidates] = True
               noise = np.logical_and(labeled_, ~goods_mask)
               outliers.append(np.nonzero(noise)[0])
               labeled_ = np.logical_and(labeled_, goods_mask) # we filter out the outliers in this step
               daal_noises[str(BETA)].append(np.sum(noise))
               print 'Number of OUTLIER queries', np.sum(noise)
               print 'Total Number of Labeled Data', np.sum(labeled_)
               #++++------- VISUALIZATION -----++++# 
               #print train_sample[0].shape
               #plot_samples(classifier, train_sample[0], samples_Px, BETA, save_to=exp_dir+'train_orders_beta={0}_{1}.png'.format(BETA,step))
               #exp_dir = exp_name+'/run={0}/'.format(iteration)
               print 'labels:', Y[candidates]
               #plot_candidates(candidates, X, save_path=exp_dir+'candidates_beta={0}_{1}.png'.format(BETA, step))
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
               train_l = (X[labeled_], to_one_hot(Y[labeled_]))
               classifier.set_weights(copy.deepcopy(weights))
               acc, predictions = evaluate(train_l, val, X, classifier, MAX_ITERS, batch_size=CLASSIFIER_BS)
               daal_accuracies[str(BETA)].append(acc)
               print "Daal accuracy BETA={2} [step={0}]: {1}".format(step, acc, BETA) 
           flog.write('step: {0}'.format(step))
           print('Daal_accuracies: {}'.format(" ".join(map(lambda x: '{0:.02f}'.format(x*100),daal_accuracies[str(BETA)]))))
           flog.write('Daal_accuracies: {0}'.format(" ".join(map(str,daal_accuracies[str(BETA)]))))
           flog.flush()
        exp_dir = exp_name+'/run={0}/'.format(iteration)
        with open(exp_dir+'accuracies.json', 'w') as f:
           json.dump(daal_accuracies, f)
        with open(exp_dir+'candidates.json', 'w') as f:
           json.dump(daal_candidates, f)
        with open(exp_dir+'num_outliers.json', 'w') as f:
           json.dump(daal_noises, f)

        plt.figure()
        handles=[]
        for BETA in BETAS:
           h,=plt.plot(range(len(daal_accuracies[str(BETA)])),daal_accuracies[str(BETA)], 'x-', label='Beta={0:.02f}'.format(BETA))    
           handles.append(h)
        plt.legend(handles=handles)
        plt.savefig(exp_name+'/run={}/al_accs.png'.format(iteration))
        daal_all.append(daal_accuracies)
        all_candidates.append(daal_candidates)
    with open(exp_name+'/all_accuracies.json', 'w') as f:
       json.dump(daal_all, f)
    with open(exp_name+'/all_candidates.json', 'w') as f:
       json.dump(all_candidates, f)
    
    flog.close()
