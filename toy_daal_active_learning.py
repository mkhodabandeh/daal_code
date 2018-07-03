import json
import sys
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from vae2d import vae
import tensorflow as tf


#################### DATA ##################
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
    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model

def evaluate(train,test,U, model, epochs=100):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    X,Y = train
    X_t, Y_t = test
    #print 'Fitting the model ...'
    model.fit(X, Y, epochs=epochs, batch_size=50, verbose=0)
    #print 'Evaluating ...'
    accuracy = model.evaluate(X_t, Y_t, verbose=0)
    #print 'Predicting ...'
    predictions = model.predict(U)
    return accuracy[1], predictions # returns 1- accuracy on validation set 2- predictions on unlabeled set

def get_boundary(model, grid):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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

 
def visualize(X, scores, candids=None, noise=None, px_map=None, filename=None, boundary=None, isShow=False,labels=None):
    plt.close('all')
    if px_map is not None: 
        fig=plt.figure()
	plt.scatter(px_map[:,0], px_map[:,1], c=px_map[:,2], s=15, cmap='autumn_r') 
        f,fmt = filename.split('.')
        plt.savefig(dirpath+f+'_map.'+fmt)
        plt.close(fig)
    fig=plt.figure()
    
    if type(scores) in [list, tuple]:
        start = (0 if labels is None else 1) 
	#fig1, axis = plt.subplots(nrows=1, ncols=len(scores)+start, figsize=(20, 6), sharey=True)       
	fig1, axis = plt.subplots(nrows=len(scores)+start, ncols=1, figsize=(3, 6), sharex=True)       
        if labels is not None:
            axis[0].scatter(X[:,0], X[:,1], c=labels, marker='.') 
            axis[0].set(adjustable='box-forced', aspect='equal')
            #axis[0].set(adjustable='box-forced')
	if boundary is not None:
            axis[start].plot(boundary[:,0], boundary[:,1], 'k^', markersize=1)
	for i in range(start,len(scores)+start):
	  handle = axis[i].scatter(X[:,0], X[:,1], c=scores[i-start], s=13, marker='.', cmap='autumn_r')
          axis[i].set(adjustable='box-forced', aspect='equal')
          #axis[i].set(adjustable='box-forced')
          fig1.colorbar(handle, ax=axis[i])
        ax = axis[-1]
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        ax = plt
	ax.scatter(X[:,0], X[:,1], c=scores,  marker='.',s=13, cmap='autumn_r')
        plt.colorbar()
    if candids is not None:
	    ax.scatter(candids[:,0], candids[:,1], marker='o', color='#00ff00', s=14)
    if noise is not None:
	    ax.scatter(noise[:,0], noise[:,1], marker='x', color='blue', s=14)
    if filename is None:
       filename = 'fig.png'
    plt.savefig(dirpath+filename)
    if isShow:
       show()
    else:
       plt.close(fig)


import os
if __name__=="__main__":
    run_id = '0' #sys.argv[1]
    exp_name = sys.argv[1]
    flog = open(exp_name+'_log_'+run_id+'.txt', 'w')
    ############# CONSTANTS ####################
    BETA =  0.0 
    dirpath='./toy_daal_plots_beta={}/'.format(BETA) 
    os.system('mkdir -p '+dirpath)
    os.system('mkdir '+exp_name)
    INITIAL_TRAINING_SIZE= 2 
    ACTIVE_LEARNING_ITERS = 5 #instead I'll run on GPU cluster, in parallel (100)
    ACTIVE_LEARNING_STEP_SIZE = 10
    ACTIVE_LEARNING_STEPS = 10 
    MAX_ITER=8000 #Classification training maximum number of iterations
    CLASS_NUM = 2 
    GRID_NUM = 250
    print '****** BETA IS : ', BETA, '********' 
    PLOT = False #True 
    ############ LOAD DATA ###############
    #clean_data = get_val(5000, True)
    try:
	    val = (np.load('val_x.npy'), np.load('val_y.npy'))
	    train = (np.load('train_x.npy'), np.load('train_y.npy' ))
    except:
            num=1000
	    val = get_val(500, True)
	    train = get_noisy_data(num, 8)
	    np.save('val_x.npy', val[0])
	    np.save('val_y.npy', val[1])
	    np.save('train_x.npy', train[0])
	    np.save('train_y.npy', train[1])
    if PLOT:
            num=500
	    plt.scatter(val[0][:num,0], val[0][:num,1], facecolors='none', edgecolors='#93DB70', s=16, )
	    plt.scatter(val[0][num:2*num,0], val[0][num:2*num,1], color='#87CEFA', s=16, marker='x')
            plt.savefig(dirpath+'data.pdf')
    if PLOT:
            plt.figure()
            num=1000
	    plt.scatter(train[0][:num,0], train[0][:num,1], facecolors='none', edgecolors='#93DB70', s=16, )
	    plt.scatter(train[0][num:2*num,0], train[0][num:2*num,1], color='#87CEFA', s=16, marker='x')
	    plt.scatter(train[0][2*num:,0], train[0][2*num:,1], color='#D3D3D3', s=6, marker='.')#, alpha=0.5)
            plt.savefig(dirpath+'data_pool.pdf')
    LEN = train[0].shape[0] 

    x_minmax = (np.min(train[0][:,0]), np.max(train[0][:,0]))
    y_minmax = (np.min(train[0][:,1]), np.max(train[0][:,1]))
    #x_minmax = np.array((-1.0, 1.0))
    #y_minmax = np.array((-1.0, 1.0))
    print GRID_NUM
    grid = get_grid(y_minmax, y_minmax, GRID_NUM)
    ############### LOAD VAE MODEL ####################
    # x   -> input
    # Px -> score
    x_inp = tf.placeholder('float32', shape=[None, 2]) 
    vae_model = vae(x_inp, n_z=2)
    #cost = vae_model['latent_loss']
    cost = vae_model['reconstr_loss']
    #cost = vae_model['cost_elem']
    saver = tf.train.Saver()
    ########## Generative Model Scores on Unlabeled/Training set ########### 
    ## It needs to be done only "once" 
    batch_size = 50 
    with tf.Session() as session:
        saver.restore(session, 'model_2d/model2d.ckpt')
        Px_val = run_all(session, cost, x_inp, train[0], batch_size=batch_size)
        #Px_val = session.run(cost, feed_dict={x: train[0]}) 
        Px_val = np.exp(-Px_val)
        print '**** Min(Px)={},  Max(Px)={} ****'.format(np.min(Px_val), np.max(Px_val))
        Px_val = Px_val.reshape((LEN,))
        #px_map = run_all(session, cost, x_inp, grid, batch_size=640)
        px_map = run_all(session, cost, x_inp, train[0], batch_size=batch_size)
	px_map= px_map.reshape((len(px_map), 1))
        print px_map.shape, grid.shape
	px_map = np.concatenate((train[0], px_map), axis=1)
	if PLOT:
		plt.figure()
		px_map[:,2]= np.power(px_map[:,2], BETA)
		plt.scatter(px_map[:,0], px_map[:,1], c=px_map[:,2], cmap='autumn_r', marker='o', s=6 ) 
		plt.savefig(dirpath+'my_px_map.png')
########## Create a classifier #####################
	print Px_val
    print np.max(Px_val), np.min(Px_val)
    classifier= get_classifier(input_dim=2, num_classes=2)
     
    #do the experiment many times with different random_seeds
    daal_all= []
    for iteration in range(ACTIVE_LEARNING_ITERS):
        print "Active Learning Iteration:", iteration
        daal_accuracies = []
        daal_num_outliers = []
        #shuffle the training data
        #X,Y,Px = copy_shuffle(train[0], train[1], Px_val)
        X, Y, Px = train[0].copy(), train[1].copy(), Px_val.copy()
        Px = normalize(Px)
        #sample the initial training set
        goods = np.nonzero(Y!=2)[0] # the reason that I'm doing this is because I don't want the initial training set to have outliers
        goods_mask = np.array([False]*LEN)
        goods_mask[goods] = True
        labeled_ = np.array([False]*LEN)
        try:
           initial_labeled = np.load('initial_{0}.npy'.forma(iteration))
        except:
           initial_labeled =  random.sample(goods, INITIAL_TRAINING_SIZE)
           #initial_labeled =  random.sample(np.nonzero(Y==0)[0], 1)+ random.sample(np.nonzero(Y==1)[0], 1)
           print 'INITIAL_LABELS', initial_labeled
           np.save('initial_{0}.npy'.format(iteration), initial_labeled)

        #np.save('initial_labeled_{}.npy'.format(iteration), initial_labeled)
        #initial_labeled = np.load('initial_labeled_{}.npy'.format(iteration))
        labeled_[initial_labeled] = True
        print('[ INITIAL LABELS ]:', Y[labeled_])
        #train the classifier with the initial set
        train_l = (X[labeled_], to_one_hot(Y[labeled_],n=2))
        #X_u = X[~labeled_] #unlabeled set
        acc, predictions = evaluate(train_l, val, X, classifier, MAX_ITER/sum(labeled_==True))
	print "Accuracy:", acc
        #### FOR TESTING PURPOSES #####
        entropies = get_entropy(predictions)#lower the better
        entropies = normalize(entropies)
        np.save('X.npy', X)
	np.save('Px.npy', Px)
	np.save('ent.npy', entropies)
        ##### DONE ###
        #arr_range = np.arange(train_data[0].shape[0])
        daal_accuracies.append(acc)
        outliers = []
        num_outliers = []
        all_noise = np.array([False]*LEN) 
        plt.figure()
        for step in range(ACTIVE_LEARNING_STEPS):
            print "step", step
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
            labeled_[candidates] = True
            noise = np.logical_and(labeled_, ~goods_mask)
            outliers.append(np.nonzero(noise)[0])
            num_outliers.append(len(outliers[-1]))
            labeled_ = np.logical_and(labeled_, goods_mask) # we filter out the outliers in this step
            print 'Number of queries for NOISY data', np.sum(noise)
            print 'Total Number of Labeled Data', np.sum(labeled_)
            #++++------- VISUALIZATION -----++++# 
            all_noise = np.logical_or(noise, all_noise)
            boundary = get_boundary(classifier,grid)
            print 'BOUNDARY', boundary.shape
            if PLOT:
		    visualize(X=X, 
				#scores=(entropies, new_Px, weighted_entropies),
				scores=(new_Px),
				filename='draw_px'+str(step)+'.png')
		    visualize(X=X, 
				#scores=(entropies, new_Px, weighted_entropies),
				scores=(entropies, weighted_entropies),
				#candids=X[candidates], 
				candids=X[candidates], 
				noise=X[noise], 
				filename='before_after_step_'+str(step)+'.png',
				boundary=boundary)
		    visualize(X=X, 
				scores=(entropies, new_Px, weighted_entropies),
				#candids=X[candidates], 
				candids=X[candidates], 
				noise=X[noise], 
				filename='step_'+str(step)+'.png',
				boundary=boundary,
				labels=Y)

		    visualize(X=X, 
				scores=weighted_entropies, 
				#candids=X[candidates], 
				candids=X[candidates], 
				noise=X[noise], 
				filename='current_step_'+str(step)+'.png',
				boundary=None,
				labels=Y)
		    visualize(X=X, 
				scores=weighted_entropies, 
				#candids=X[candidates], 
				candids=X[labeled_], 
				noise=X[all_noise], 
				filename='all_until_step_'+str(step)+'.png',
				boundary=None,
				labels=Y)
            #++++------- VISUALIZATION -----++++# 
            #traind and test
            train_l = (X[labeled_], to_one_hot(Y[labeled_],n=2))
            acc, predictions = evaluate(train_l, val, X, classifier, MAX_ITER/sum(labeled_==True))
            daal_accuracies.append(acc)
            print "Daal accuracy [step={0}]: {1}".format(step, acc) 
        if PLOT:
		plt.figure()
		plt.plot(range(len(daal_accuracies)),daal_accuracies, 'xb-')    
		plt.savefig(dirpath+'al_accs_{}.png'.format(iteration))
        daal_all.append(daal_accuracies[:])
        #print('Daal_accuracies: {}'.format("".join(map(str,daal_accuracies))))
        with open(exp_name+'/accuracies_run={}.json'.format(iteration), 'w') as f:
           json.dump(daal_accuracies, f)
        with open(exp_name+'/num_outliers={}.json'.format(iteration), 'w') as f:
           json.dump(num_outliers, f)
        #np.save('accuracies_beta={0}_run={1}.npy'.format(BETA,iteration), daal_accuracies)
        #np.save('num_noises_beta={0}_run{1}.npy'.format(BETA,iteration), map(lambda x: len(x),outliers) )
        flog.write('step: {0}'.format(step))
        #flog.write('Daal_accuracies: {0}'.format("".join(map(str,daal_accuracies))))
        flog.flush()
    flog.close()
    with open(exp_name+'/all_accuracies.json', 'w') as f:
           json.dump(daal_all, f)
    
