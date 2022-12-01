import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import os
import sys
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.model_selection import train_test_split

tf.reset_default_graph()

# SET ENV
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
config.gpu_options.allow_growth=True


def fc_bn(_x, _output, _phase, _scope):
	with tf.variable_scope(_scope):
		h1 = tf.contrib.layers.fully_connected(_x, _output, activation_fn=None, scope='dense', weights_initializer=tf.contrib.layers.variance_scaling_initializer(), weights_regularizer = tf.contrib.layers.l2_regularizer(0.01), reuse=tf.AUTO_REUSE)
		h2 = tf.contrib.layers.batch_norm(h1, updates_collections=None, fused=True, decay=0.9, center=True, scale=True, is_training=_phase, scope='bn', reuse=tf.AUTO_REUSE)
		return h2

allergy = sys.argv[1] 
group = sys.argv[2] 
x_data_file = sys.argv[3]
y_data_file = sys.argv[4]
x_test_file = sys.argv[5]
y_test_file = sys.argv[6]

x_data = pd.read_csv(x_data_file) #Row: each sample, Column: Features
y_data = pd.read_csv(y_data_file)
x_test = pd.read_csv(x_test_file)
y_test = pd.read_csv(y_test_file)

num_timepoint = 8 #len(x[0])
num_feature = len(x_data.columns)
x_data = x_data.values
x_data = x_data.reshape(-1, num_timepoint, num_feature)
num_samples = len(x_data)

x_test = x_test.values
x_test = x_test.reshape(-1, num_timepoint, num_feature)

n_classes = 2

tf_X = tf.placeholder(tf.float32, [None, num_timepoint, num_feature])
tf_Y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool, name='phase')

def build_discriminator(X, _phase, _keep_prob) :
	bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32), merge_mode = 'concat')(X)
	fc1 = tf.nn.dropout(tf.nn.leaky_relu(fc_bn(bilstm, 16, _phase, "discriminator_fc1")), _keep_prob)
	logits = fc_bn(fc1, 1, _phase, "logits")
	predicted_value = tf.nn.softmax(logits)
	return predicted_value, logits 


d_pred, d_logits = build_discriminator(tf_X, phase, keep_prob)
d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = d_logits, labels = tf_Y))

learning_rate_d = 0.001
num_epoch = 4000

d_train_step = tf.train.AdamOptimizer(learning_rate_d).minimize(d_loss)

pred = tf.argmax(d_pred, 1)
label = tf.argmax(tf_Y, 1)
correct_pred = tf.equal(pred, label)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
_accuracy = tf.Variable(0)

max_acc = 0.0
dp_rate = 0.5

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(num_epoch):
		_, d_loss_print, d_acc = sess.run([d_train_step, d_loss, accuracy], feed_dict={tf_X: x_data, tf_Y: y_data, phase : True, keep_prob: dp_rate})
		if i % 10 == 0:
			test_acc, test_pred, test_label = sess.run([accuracy, pred, label], feed_dict={tf_X: x_test, tf_Y: y_test, phase : False, keep_prob: 1.0})
			print('Epoch: %d, cost: %f, train_acc:%.4f, test_acc: %.4f' % (i, d_loss_print, d_acc, test_acc))
			if test_acc > max_acc :
				max_acc = test_acc
				max_pred = test_pred
				max_label = test_label
	np.savetxt("prediction_group_" + group + "_" + allergy + "_acc_" + str(max_acc) + ".csv",  max_pred, fmt="%.0f", delimiter=",")
	np.savetxt("label_group_" + group + "_" + allergy + "_acc_" + str(max_acc) + ".csv",  max_label, fmt="%.0f", delimiter=",")

				
