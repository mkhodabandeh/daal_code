import numpy

import os
import urllib
import gzip
import cPickle as pickle

def get_mnist():
    filepath = 'mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)
        return train_data, dev_data, test_data	
   
def mnist_generator(data, batch_size, n_labelled, limit=None, portions=[0, 1], labels=[1,1,1,1,1,1,1,1,1,1]):
    images, targets = data
    print 'IMAGES.SHAPE', images.shape
    print 'before slicing: len(images)', len(images), 'len(targets)', len(targets) , targets.shape
    new_images, new_targets = numpy.zeros((0,784)), numpy.zeros((0, )) 
    start_ind = int(len(images)*portions[0]/1.0)
    end_ind = int(len(images)*portions[1]/1.0)
    for i in range(10):
       inds = targets == i
       l = targets[inds]
       l = l[start_ind:end_ind]
       #new_targets.extend(l[0:int(len(l)*labels[i]/1.0)])
       new_targets = numpy.concatenate((new_targets,l[0:int(len(l)*labels[i]/1.0)]), 0)
       ims = images[inds]
       ims = ims[start_ind:end_ind]
       new_images = numpy.concatenate((new_images,ims[0:int(len(l)*labels[i]/1.0)]), 0)
    m = (len(new_images)/batch_size)*batch_size
    images = new_images[0:m]
    targets = new_targets[0:m]
    print 'len(images)', images.shape, 'len(targets)', targets.shape
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)
    if limit is not None:
        print "WARNING ONLY FIRST {} MNIST DIGITS".format(limit)
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        #print 'images.shape', images.shape
        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None, portions=[0, 1], labels=[1]*10):
    filepath = 'mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    
    return (
        mnist_generator(train_data, batch_size, n_labelled, portions=portions, labels=labels), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)

    )
