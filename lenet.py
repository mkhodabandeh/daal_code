import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected, reshape
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np

class Lenet:

	def __init__(self, session=None, scope=''):
                self.n_epoch = 50 
                self.num_classes= 10
		self.batch_size = 128
                if session is None:
		    print "New Session -> Lenet"
                    new_graph = tf.Graph()
                    self.sess = tf.Session(graph=new_graph)
                    self.g = new_graph
                else:
                    self.sess = session
                    self.g = tf.get_default_graph() 
                    #self.g= tf.Graph()
                self.scope = scope
		self.model = self.create_model()
		#try:
		#    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		#    self.sess.run(tf.variables_initializer(tf.get_collection_ref('is_training')))
		#except:
		#    init = tf.initialize_all_variables()
		var_list = [v for v in tf.global_variables() if scope in v.name ]
		#print var_list
		init = tf.initialize_variables(var_list)
		self.sess.run(init)

	def create_model(self):
           with self.g.as_default():
	     with tf.variable_scope(self.scope):
		#network = input_data(shape=[None, 784], name='lenet_input')
                #network = reshape(network, [-1, 28, 28, 1], name='lenet_input')
		network = input_data(shape=[None, 28, 28, 1], name='lenet_input')
		network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
		network = max_pool_2d(network, 2)
		network = local_response_normalization(network)
		network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
		network = max_pool_2d(network, 2)
		network = local_response_normalization(network)
		network = fully_connected(network, 128, activation='tanh', name='fc1')
		# self.w = tflearn.variables.get_layer_variables_by_name('fc1')
		network = dropout(network, 0.8 )
		network = fully_connected(network, 256, activation='tanh')
		network = dropout(network, 0.8)
		network = fully_connected(network, 10, activation='softmax')
                network = regression(network, optimizer='adam', learning_rate=0.01,
                                                    loss='categorical_crossentropy', name='target')
		model = tflearn.DNN(network,  session=self.sess)
		return model

	def train(self, data,n_epoch, restart=True):
            print "TRAINING", self.scope
            n_epoch = max(1, n_epoch)
            n_epoch = min(2000, n_epoch)
            if restart:
                optimizer_vars = [v for v in tf.global_variables() if 'Adam' in v.name]
                self.sess.run(tf.initialize_variables(optimizer_vars))
	    with self.sess.as_default():
              with self.g.as_default():
		images,targets = data
	        with tf.variable_scope(self.scope):
		   self.model.fit({self.scope+'/lenet_input': images}, {self.scope+'/target': targets}, n_epoch=n_epoch, batch_size=self.batch_size, snapshot_step=100, show_metric=True, run_id='convnet_mnist_'+self.scope)

	def evaluate(self, data):
	    with self.sess.as_default():
              with self.g.as_default():
                x_test, y_test = data
		score = self.model.evaluate(x_test, y_test, batch_size=self.batch_size)
		return score

	def predict(self, data):
          with self.g.as_default():
            x_train = data
	    batch_size = self.batch_size
	    with self.sess.as_default():
		arr = np.zeros((x_train.shape[0], self.num_classes))
                # print arr.shape
		batch_size = 128
		for i in xrange(x_train.shape[0]/self.batch_size):
		    # arr[i*batch_size:(i+1)*batch_size,...] =  
		    temp = np.array(self.model.predict({self.scope+'/lenet_input': x_train[i*batch_size:(i+1)*batch_size, ...]}))
		    # print temp.shape
		    arr[i*batch_size:(i+1)*batch_size, ...] = temp
		    # print i*batch_size, (i+1)*batch_size
		    # print temp.shape
		    # print temp[0:2, ...]
		    # pdb.set_trace()
		# print (i+1)*batch_size, self.x_train.shape[0]
		i = x_train.shape[0]/batch_size
		if (i)*batch_size != x_train.shape[0]:
		    # print 'hi'
		    temp = np.array(self.model.predict({self.scope+'/lenet_input': x_train[(i)*batch_size:, ...]}))
		    arr[(i)*batch_size:, ...] = temp

		# return np.array(self.model.predict({'lenet_input': self.x_train}))
		return arr


if __name__=='__main__':
   sess = tf.Session()
   lenet = Lenet(sess, scope='hi')
   import keras
   from keras.datasets import mnist
   (x, y), (x_test, y_test) = mnist.load_data()
   x = (x.astype(np.float32) - 127.5)/127.5
   x = x[:, :, :, None]
   x_test = (x_test.astype(np.float32) - 127.5)/127.5
   x_test = x_test[:, :, :, None]
   y = keras.utils.to_categorical(y, num_classes=10)
   y_test = keras.utils.to_categorical(y_test, num_classes=10)
   indices = range(300) 
   lenet.train((x[indices], y[indices]))
   score1 = lenet.evaluate((x_test, y_test))
   print score1
