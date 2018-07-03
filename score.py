import numpy as np
from gan_mnist import *
from tflib.mnist import get_mnist
import tflib as lib
import tflib.save_images
if __name__=="__main__":
    ACTIVE_LEARNING_ITERS = 100
    #saver = tf.train.Saver()
    print 'model', MODEL_PATH
    BATCH_SIZE=64
    g = tf.Graph()
    with tf.Session(graph=g) as session:
        real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
        disc_real = Discriminator(real_data)
	#saver = tf.train.import_meta_graph(MODEL_PATH+'/my-model-100.meta')
        var_list = [v for v in tf.global_variables() if  "Discriminator" in v.name ]
        saver = tf.train.Saver(var_list)
        print '**** LOADING FROM', tf.train.latest_checkpoint(MODEL_PATH)
	saver.restore(session, tf.train.latest_checkpoint(MODEL_PATH))
        test_accuracies = []
        train_data, dev_data, test_data = get_mnist()

        images, labels = train_data
        images = images[len(images)/2:]
        labels = labels[len(images)/2:]
        print 'labels'
        print labels[:BATCH_SIZE].reshape((8,8))
	print images[:10].shape
        for i in range(10):
                ims = images[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                print 'SHAPE -> ', ims.shape
		disc_scores = session.run(disc_real, feed_dict={real_data: ims})
		inds = np.argsort(disc_scores[0])[::-1]
		lib.save_images.t_save_images(
		    ims.reshape((BATCH_SIZE, 28, 28)), 
		    'results_{}_org.png'.format(i),
		    (1, BATCH_SIZE)
		)
		lib.save_images.t_save_images(
		    ims[inds].reshape((BATCH_SIZE, 28, 28)), 
		    'results_{}.png'.format(i),
		    (1, BATCH_SIZE)
		)
                print i 
