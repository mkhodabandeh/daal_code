from gan_mnist import *
from tflib.mnist import get_mnist
from lenet import Lenet
import keras
import numpy as np
import random
def get_entropy(preds):
    return np.sum(preds*np.log(preds), axis=1) 
def run_all(session, network, inp, data):
    with session.as_default():
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

if __name__=="__main__":
    run_id = sys.argv[1]
    flog = open('logS'+run_id+'.txt', 'w')
    INITIAL_TRAINING_SIZE=20
    ACTIVE_LEARNING_ITERS = 1 #instead I'll run on GPU cluster, in parallel (100)
    ACTIVE_LEARNING_STEP_SIZE = 10
    ACTIVE_LEARNING_STEPS = 100
    CLASS_NUM = 10
    new_graph = tf.Graph()
    #saver = tf.train.import_meta_graph(MODEL_PATH+'/model-100.meta')
    print MODEL_PATH
    with tf.Session(graph=new_graph) as session:
        # Create and Restore pre-trained  GAN 
        real_data = tf.placeholder(tf.float32, shape=[128, OUTPUT_DIM])
        disc_real = Discriminator(real_data)
        var_list = [v for v in tf.global_variables() if  "Discriminator" in v.name ]
        saver = tf.train.Saver(var_list)
	saver.restore(session, tf.train.latest_checkpoint(MODEL_PATH))
        # get GAN discriminator score for training set
        train_data, dev_data, test_data = get_mnist()
	disc_scores = run_all(session, disc_real, real_data, train_data[0])
	disc_scores -= min(disc_scores)
	disc_scores /= max(disc_scores) 
        # Create a classifier
        classifier_entropy = Lenet(None,scope='entropy_lenet')
        classifier_daal = Lenet(None,scope='daal_lenet')
        var_list = [v for v in tf.global_variables() if  "target" in v.name ]
        print "*************"
	print var_list
        var_list = [v for v in tf.global_variables() if  "lenet" in v.name ]
        print "*************"
	for v in var_list:
		print v
        train_data = (train_data[0], keras.utils.to_categorical(train_data[1], num_classes=CLASS_NUM)) 
        test_data = (test_data[0], keras.utils.to_categorical(test_data[1], num_classes=CLASS_NUM)) 
        #do the experiment many times with different random_seeds
        daal_all= []
        entropy_all = []
        for iteration in range(ACTIVE_LEARNING_ITERS):
            print "Active Learning Iteration:", iteration
            daal_accuracies = []
            entropy_accuracies = []
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
            train_set_indices = random.sample(range(train_data[0].shape[0]), INITIAL_TRAINING_SIZE)
            score_argsort = np.argsort(scores[:]) 
            daal_set_indices = score_argsort[-INITIAL_TRAINING_SIZE:].tolist()
            print 'TRAIN_SET_INDICES', len(train_set_indices)
            print 'DAAL_SET_INDICES', len(daal_set_indices)
            #train the classifier with the initial set
            my_train_data = (images[train_set_indices], targets[train_set_indices])
            daal_train_data = (images[daal_set_indices], targets[daal_set_indices])
            classifier_daal.train(daal_train_data)
            classifier_entropy.train(my_train_data)
            acc = classifier_entropy.evaluate(test_data)
            entropy_accuracies.append(acc)
            acc = classifier_daal.evaluate(test_data)
            daal_accuracies.append(acc)
           
            for step in range(ACTIVE_LEARNING_STEPS):
                print "step", step
                ## Entropy: 
                #get predicted probabilities of the unlabel set and compute the entropies
                print "making entropy decisions"
                predictions = classifier_entropy.predict(images[train_set_indices]) 
                entropies = get_entropy(predictions)#lower the better
                argsorted = np.argsort(entropies)[::-1]#increasing sort
                #start from the beginning and find the first 10 samples that are not already labelled
                candidates = []
		i = 0
                while len(candidates)<ACTIVE_LEARNING_STEP_SIZE:
                    if argsorted[i] not in train_set_indices:
                        candidates.append(argsorted[i])
                    i+=1
                #merge training set with the labelled 
                train_set_indices+=candidates
                #traind and test
                classifier_entropy.train((images[train_set_indices], targets[train_set_indices]))
                acc = classifier_entropy.evaluate(test_data)
                entropy_accuracies.append(acc)
                print "Entropy accuracy [step={0}]: {1}".format(step, acc) 
                ## Distribution Aware
                predictions = classifier_daal.predict(images[daal_set_indices]) 
                entropies = get_entropy(predictions)#lower the better
                weighted_entropies = entropies*scores[daal_set_indices]
                argsorted = np.argsort(weighted_entropies)[::-1]#increasing sort
                candidates = []
                i = 0
                while len(candidates)<ACTIVE_LEARNING_STEP_SIZE:
                    if argsorted[i] not in daal_set_indices:
                        candidates.append(argsorted[i])
                    i+=1
                daal_set_indices+=candidates
                #traind and test
                classifier_daal.train((images[daal_set_indices], targets[daal_set_indices]))
                acc = classifier_daal.evaluate(test_data)
                daal_accuracies.append(acc)
                print "Daal accuracy [step={0}]: {1}".format(step, acc) 

        daal_all.append(daal_accuracies[:])
        entropy_all.append(entropy_accuracies[:])
        print('Entropy_accuracies: {0}'.format("".join(map(str,entropy_accuracies))))
        print('Daal_accuracies: {0}'.format("".join(map(str,daal_accuracies))))
        flog.write('step: {0}'.format(step))
        flog.write( 'Entropy_accuracies: {0}\n'.format("".join(map(str,entropy_accuracies))))
        flog.write('Daal_accuracies: {0}'.format("".join(map(str,daal_accuracies))))
    flog.close()
