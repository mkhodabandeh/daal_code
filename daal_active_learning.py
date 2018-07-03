from sklearn import preprocessing
import tensorflow as tf
import sys
from dcgan import *
from lenet import Lenet
import keras
import numpy as np
import random
from tflib.mnist import get_mnist

def get_entropy(preds):
    return np.sum(-1*preds*np.log(preds), axis=1) 

def run_all(session, network, inp, data):
    batch_size=128
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

def normalize(arr):
    arr = arr.copy() - np.min(arr)
    arr = arr / np.max(arr)
    return arr
 
if __name__=="__main__":
    run_id = sys.argv[1]
    exp_name = sys.argv[2]
    flog = open(exp_name+'_log_'+run_id+'.txt', 'w')
    INITIAL_TRAINING_SIZE=20
    ACTIVE_LEARNING_ITERS = 1 #instead I'll run on GPU cluster, in parallel (100)
    ACTIVE_LEARNING_STEP_SIZE = 10
    ACTIVE_LEARNING_STEPS = 100
    MAX_ITER=8000 #Classification training maximum number of iterations
    CLASS_NUM = 10
    new_graph = tf.Graph()
    #saver = tf.train.import_meta_graph(MODEL_PATH+'/model-100.meta')
    with tf.Session(graph=new_graph) as session:
        # Create and Restore pre-trained  GAN 
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        #load and prepare the data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        X_train = X_train[:, :, :, None]
        X_test= (X_test.astype(np.float32) - 127.5)/127.5
        X_test = X_test[:, :, :, None]
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
        train_data = (train_data[0], keras.utils.to_categorical(train_data[1], num_classes=CLASS_NUM)) 
        test_data = (test_data[0], keras.utils.to_categorical(test_data[1], num_classes=CLASS_NUM)) 
        # get GAN discriminator score for training set
        d_pret = d.predict(X_train, verbose=1)
        d_pret = d_pret.reshape((60000,))
        #d_pret = np.sort(d_pret)
        disc_scores = preprocessing.scale(d_pret) + 0.5
        # Create a classifier
        classifier_daal = Lenet(session,scope='daal_lenet')
        #do the experiment many times with different random_seeds
        daal_all= []
        for iteration in range(ACTIVE_LEARNING_ITERS):
            print "Active Learning Iteration:", iteration
            daal_accuracies = []
            #shuffle the training data
            images,targets = train_data[0].copy(), train_data[1].copy() #TODO better not to copy and play with indices, instead. whatever!
            scores = disc_scores[:]
            assert len(scores) == train_data[0].shape[0]
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(targets)
            np.random.set_state(rng_state)
            np.random.shuffle(scores)
            #sample the initial training set
            daal_set_indices = random.sample(range(train_data[0].shape[0]), INITIAL_TRAINING_SIZE)
            #train the classifier with the initial set
            daal_train_data = (images[daal_set_indices], targets[daal_set_indices])
            classifier_daal.train(daal_train_data, MAX_ITER/len(daal_train_data))
            acc = classifier_daal.evaluate(test_data)
	    #arr_range = np.arange(train_data[0].shape[0])
            daal_accuracies.append(acc)
            for step in range(ACTIVE_LEARNING_STEPS):
                print "step", step
                #get predicted probabilities of the unlabel set and compute the entropies
                predictions = classifier_daal.predict(images) 
                entropies = get_entropy(predictions)#lower the better
		entropies = normalize(entropies)
                #entropies -> less better [0-1]
                entropies = 1-entropies
                weighted_entropies = entropies*scores
		# we would like to select the largest
                argsorted = np.argsort(weighted_entropies)[::-1]#increasing sort, then reversed 
                #argsorted = arr_range[daal_set_indices][argsorted]
                candidates = []
                i = 0
                while len(candidates)<ACTIVE_LEARNING_STEP_SIZE:
                    if argsorted[i] not in daal_set_indices:
                        candidates.append(argsorted[i])
                    i+=1
                daal_set_indices+=candidates
                #traind and test
                classifier_daal.train((images[daal_set_indices], targets[daal_set_indices]), MAX_ITER/len(daal_set_indices))
                acc = classifier_daal.evaluate(test_data)
                daal_accuracies.append(acc)
                print "Daal accuracy [step={0}]: {1}".format(step, acc) 

        daal_all.append(daal_accuracies[:])
        print('Daal_accuracies: {0}'.format("".join(map(str,daal_accuracies))))
        flog.write('step: {0}'.format(step))
        flog.write('Daal_accuracies: {0}'.format("".join(map(str,daal_accuracies))))
        flog.flush()
    flog.close()
