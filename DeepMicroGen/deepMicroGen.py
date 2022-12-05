import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

tf.reset_default_graph()

# SET ENV
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
config.gpu_options.allow_growth=True


def fc_bn(_x, _output, _phase, _scope):
	with tf.variable_scope(_scope):
		h1 = tf.contrib.layers.fully_connected(_x, _output, activation_fn=None, scope='dense', weights_initializer=tf.contrib.layers.variance_scaling_initializer(), weights_regularizer = tf.contrib.layers.l2_regularizer(0.01), reuse=tf.AUTO_REUSE)
		h2 = tf.contrib.layers.batch_norm(h1, updates_collections=None, fused=True, decay=0.9, center=True, scale=True, is_training=_phase, scope='bn', reuse=tf.AUTO_REUSE)
		return h2

def softmax(x) :
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

profiles_filename = sys.argv[1]
mask_filename = sys.argv[2]
learning_rate_input = float(sys.argv[3])
dropout_rate_input = float(sys.argv[4])
epochs_input = int(sys.argv[5])
pseudo_count = float(sys.argv[6])

x_data = pd.read_csv(profiles_filename, index_col = 0) #Row: each sample, Column: Features
cluster_num = len(x_data["cluster"].unique().tolist())
cluster_df_list = []
for i in range(cluster_num) :
	tmp_cluster = pd.DataFrame(x_data[x_data["cluster"] == i])
	del tmp_cluster["cluster"]
	tmp_cluster = tmp_cluster.T
	cluster_df_list.append(tmp_cluster)

mask_data = pd.read_csv(mask_filename, index_col = 0) #Row: sample, Column: timepoint
mask_data.columns = mask_data.columns.astype('int')
num_timepoint = len(mask_data.columns)
mask_vector = mask_data.values
mask_vector = mask_vector.reshape(-1, num_timepoint, 1)

timepoints = mask_data.columns.tolist()
time_list = []
for i in range(len(mask_data)) :
        time_list.extend(timepoints)

time_df = pd.DataFrame({'month' : time_list})
time_label_input = pd.get_dummies(time_df['month'])
timelist = time_label_input.columns.astype('float').tolist()
time_label = time_label_input.values
time_label = time_label.reshape(-1, num_timepoint, num_timepoint)

del x_data["cluster"]
x_data = x_data.T
num_feature = len(x_data.columns)

x = x_data.values
x = x.reshape(-1, num_timepoint, num_feature)
num_samples = len(x)

cluster_feature_num_list = []
for i in range(cluster_num) :
	cluster_feature_num_list.append(len(cluster_df_list[i].columns))
	cluster_df_list[i] = cluster_df_list[i].values
	cluster_df_list[i] = cluster_df_list[i].reshape(-1, num_timepoint, cluster_feature_num_list[i])

mask_data_tmp = mask_data.T # Row: timepoint, Column: sample
mask_data_tmp.reset_index(inplace = True, drop = True)
sample_list = mask_data_tmp.columns.tolist()
mask_data_tmp['time'] = timelist

dict_timedecay_f = {}
dict_timedecay_b = {}

for sample in sample_list :
	dict_timedecay_f[sample] = []
	dict_timedecay_b[sample] = []

for sample in sample_list :
	for time in range(num_timepoint) :
		if time == 0 :
			tmp = 0
			dict_timedecay_f[sample].append(0)
		else :
			if mask_data_tmp[sample][time-1] == 1 :
				tmp = mask_data_tmp['time'][time] - mask_data_tmp['time'][time-1]
				dict_timedecay_f[sample].append(tmp)
			else :
				tmp += mask_data_tmp['time'][time] - mask_data_tmp['time'][time-1]
				dict_timedecay_f[sample].append(tmp)


for sample in sample_list :
	for time in range(num_timepoint-1, -1, -1) :
		if time == num_timepoint-1 :
			tmp = 0
			dict_timedecay_b[sample].append(0)
		else :
			if mask_data_tmp[sample][time+1] == 1 :
				tmp = abs(mask_data_tmp['time'][time] - mask_data_tmp['time'][time+1])
				dict_timedecay_b[sample].append(tmp)
			else :
				tmp += abs(mask_data_tmp['time'][time] - mask_data_tmp['time'][time+1])
				dict_timedecay_b[sample].append(tmp)

dict_timedecay_f = pd.DataFrame(dict_timedecay_f)
dict_timedecay_f = dict_timedecay_f.T
timeDecay_forward = dict_timedecay_f.values.reshape(-1, num_timepoint, 1)

dict_timedecay_b = pd.DataFrame(dict_timedecay_b)
dict_timedecay_b = dict_timedecay_b.T
timeDecay_backward = dict_timedecay_b.values.reshape(-1, num_timepoint, 1)


tf_X = tf.placeholder(tf.float32, [None, num_timepoint, num_feature])
tf_Y_time = tf.placeholder(tf.float32, [None, num_timepoint, num_timepoint])
tf_mask = tf.placeholder(tf.float32, [None, num_timepoint, 1])
tf_timeDecay_f = tf.placeholder(tf.float32, [None, num_timepoint, 1])
tf_timeDecay_b = tf.placeholder(tf.float32, [None, num_timepoint, 1])
phase_gen = tf.placeholder(tf.bool, name='phase_gen')
phase_discriminator = tf.placeholder(tf.bool, name='phase_discriminator')
keep_prob = tf.placeholder(tf.float32)

tf_X_cluster_list = []
for i in range(len(cluster_df_list)) :
	tf_X_cluster_list.append(tf.placeholder(tf.float32, [None, num_timepoint, cluster_feature_num_list[i]]))

def cnn_layer(_x, _num_feature, _scope, _keep_prob):
	with tf.variable_scope(_scope):
		cnn_1 = tf.nn.dropout(tf.keras.layers.Conv1D(16, 3, strides = 1, padding = 'same', input_shape = (num_timepoint, _num_feature), activation = tf.nn.leaky_relu)(_x), _keep_prob)
		cnn_1_maxpool = tf.keras.layers.MaxPool1D(pool_size = 1, strides = 1, padding = 'same')(cnn_1)
		cnn_2 = tf.nn.dropout(tf.keras.layers.Conv1D(8, 3, strides = 1, padding = 'same', input_shape = (num_timepoint, 16), activation = tf.nn.leaky_relu)(cnn_1_maxpool), _keep_prob)
		cnn_2_maxpool = tf.keras.layers.MaxPool1D(pool_size = 1, strides = 1, padding = 'same')(cnn_2)
		return cnn_2_maxpool

def build_generator(input_x, cluster_x, mask, timedecay_x_f, timedecay_x_b, _phase, _keep_prob) :
	for i in range(len(cluster_x)) :
		if i == 0 :
			cluster_cnn_feature = cnn_layer(cluster_x[i], cluster_feature_num_list[i], "cnn_cluster_" + str(i), _keep_prob)
		else :
			tmp_feature = cnn_layer(cluster_x[i], cluster_feature_num_list[i], "cnn_cluster_" + str(i), _keep_prob)
			cluster_cnn_feature = tf.concat([cluster_cnn_feature, tmp_feature], 2)
	
	rnn_f, rnn_b = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(10, return_sequences = True), merge_mode = None)(cluster_cnn_feature)
	output_rnn_f = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_feature))(rnn_f)
	output_rnn_b = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_feature))(rnn_b)
	difference_fb = tf.abs(tf.subtract(output_rnn_f, output_rnn_b))
	#
	timedecay_f_fc = tf.nn.leaky_relu(fc_bn(timedecay_x_f, 1, _phase, "timdecay_f_fc"))
	timedecay_f_max = tf.maximum(tf.zeros((num_samples, num_timepoint, 1)), timedecay_f_fc)
	#timedecay_f_max = tf.maximum(tf.zeros((num_samples, num_timepoint, 1)), timedecay_x_f)
	timedecay_f_max = -1 * timedecay_f_max
	lamda_f = tf.exp(timedecay_f_max)
	#
	timedecay_b_fc = tf.nn.leaky_relu(fc_bn(timedecay_x_b, 1, _phase, "timdecay_b_fc"))
	timedecay_b_max = tf.maximum(tf.zeros((num_samples, num_timepoint, 1)), timedecay_b_fc)
	#timedecay_b_max = tf.maximum(tf.zeros((num_samples, num_timepoint, 1)), timedecay_x_b)
	timedecay_b_max = -1 * timedecay_b_max
	lamda_b = tf.exp(timedecay_b_max)
	#
	x_estimated = tf.add(tf.multiply(lamda_f,output_rnn_f), tf.multiply(lamda_b, output_rnn_b))
	x_real_mask = tf.multiply(input_x, mask)
	x_estimated_mask = tf.multiply(x_estimated, tf.subtract(tf.ones((num_samples, num_timepoint, 1)), mask))
	#
	x_imputed = tf.add(x_real_mask, x_estimated_mask)
	return x_estimated, x_imputed , x_real_mask, x_estimated_mask, difference_fb

def build_discriminator(X, _phase) :
	lstm = tf.keras.layers.LSTM(10, return_sequences = True)(X)
	sm_out = tf.nn.softmax(fc_bn(lstm, num_timepoint, _phase, "logits_label"))
	output_lstm2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))(lstm)
	logits = fc_bn(output_lstm2, 1, _phase, "logits")
	predicted_value = tf.nn.sigmoid(logits)
	return predicted_value, logits, sm_out 

generated_x, imputed_x, x_real_mask, x_estimated_mask, difference_fb = build_generator(tf_X, tf_X_cluster_list, tf_mask, tf_timeDecay_f, tf_timeDecay_b, phase_gen, keep_prob)
D_real_pred, D_real_logits, D_real_logits_label = build_discriminator(x_real_mask, phase_discriminator)
D_fake_pred, D_fake_logits, D_fake_logits_label = build_discriminator(x_estimated_mask, phase_discriminator)

d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_logits, labels = tf.ones_like(D_real_logits)))
d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits, labels = tf.zeros_like(D_fake_logits)))
disc_loss = d_real_loss + d_fake_loss
time_class_loss = tf.reduce_mean(-tf.reduce_sum(tf_Y_time * tf.log(D_real_logits_label + 1e-10), axis = 1))
d_loss = disc_loss + time_class_loss

gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits, labels = tf.ones_like(D_fake_logits)))
masked_reconstruction_loss = tf.reduce_mean(tf.abs(x_real_mask - tf.multiply(generated_x, tf_mask)))
consistency_loss = tf.reduce_mean(difference_fb)

g_loss = gen_loss + masked_reconstruction_loss + consistency_loss

learning_rate_d = learning_rate_input
learning_rate_g = learning_rate_input

d_train_step = tf.train.AdamOptimizer(learning_rate_d).minimize(d_loss)
g_train_step = tf.train.AdamOptimizer(learning_rate_g).minimize(g_loss)

num_discriminator_epoch = 5
num_generator_epoch = 1
num_epoch = epochs_input
min = 100
keep_rate = dropout_rate_input
stop_count = 0

gen_feed_dict = {tf_X: x, tf_mask: mask_vector, tf_timeDecay_f : timeDecay_forward, tf_timeDecay_b : timeDecay_backward, phase_discriminator : False, phase_gen: True, keep_prob: keep_rate, tf_Y_time : time_label}
for j in range(len(cluster_df_list)) :
	gen_feed_dict[tf_X_cluster_list[j]] = cluster_df_list[j]

imp_feed_dict = {tf_X: x, tf_mask: mask_vector, tf_timeDecay_f : timeDecay_forward, tf_timeDecay_b : timeDecay_backward, phase_discriminator : False, phase_gen: False, keep_prob: 1.0, tf_Y_time : time_label}
for j in range(len(cluster_df_list)) :
	imp_feed_dict[tf_X_cluster_list[j]] = cluster_df_list[j]

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(10) :
		_, g_loss_print = sess.run([g_train_step, g_loss], feed_dict=gen_feed_dict)
	for i in range(num_epoch):
		for j in range(num_discriminator_epoch) :
			imputed_dataset = sess.run(imputed_x, feed_dict = imp_feed_dict)
			disc_feed_dict = {tf_X: imputed_dataset, tf_mask: mask_vector, tf_timeDecay_f : timeDecay_forward, tf_timeDecay_b : timeDecay_backward, phase_discriminator : True, phase_gen: False, keep_prob: 1.0, tf_Y_time : time_label}
			for t in range(len(cluster_df_list)) :
				disc_feed_dict[tf_X_cluster_list[t]] = cluster_df_list[t]
			_, d_loss_print = sess.run([d_train_step, d_loss], feed_dict=disc_feed_dict)
		for j in range(num_generator_epoch) :
			imputed_dataset = sess.run(imputed_x, feed_dict = imp_feed_dict)
			gen_feed_dict = {tf_X: imputed_dataset, tf_mask: mask_vector, tf_timeDecay_f : timeDecay_forward, tf_timeDecay_b : timeDecay_backward, phase_discriminator : False, phase_gen: True, keep_prob: keep_rate, tf_Y_time : time_label}
			for t in range(len(cluster_df_list)) :
				gen_feed_dict[tf_X_cluster_list[t]] = cluster_df_list[t]
			_, g_loss_print, r_loss, c_loss = sess.run([g_train_step, g_loss, masked_reconstruction_loss, consistency_loss], feed_dict=gen_feed_dict)
		if i % 10 == 0:
			print('Epoch: %d, g_loss: %f, r_loss:%f, c_loss: %f, d_loss: %f' % (i, g_loss_print, r_loss, c_loss, d_loss_print))
			if min > r_loss :
				min = r_loss
				stop_count = 0
				final_imputed_dataset = imputed_dataset 
		stop_count += 1
		if stop_count == 1000 :
			break
	imputed_dataset_reshape = final_imputed_dataset.reshape(-1, num_feature)
	imputed_dataset_reshape_df = pd.DataFrame(imputed_dataset_reshape)
	imputed_dataset_reshape_df.columns = x_data.columns
	imputed_dataset_reshape_df.index = x_data.index 
	imputed_dataset_reshape_df = imputed_dataset_reshape_df.T 
	imputed_dataset_reshape_df.to_csv("imputed_dataset_from_DeepMicroGen_clr.csv", mode = "w", index = True)
	imputed_dataset_reshape_df_scaled = softmax(imputed_dataset_reshape_df)
	sample_list = imputed_dataset_reshape_df_scaled.columns
	for sample in sample_list :
		for j in range(len(imputed_dataset_reshape_df_scaled)) :
			if imputed_dataset_reshape_df_scaled[sample][j] < pseudo_count :
				imputed_dataset_reshape_df_scaled[sample][j] = 0
	for sample in sample_list :
		tmp_diff = 1 - imputed_dataset_reshape_df_scaled[sample].sum()
		tmp_sample = imputed_dataset_reshape_df_scaled[sample]
		non_zero_count = len(tmp_sample[tmp_sample != 0.0])
		add_pseudo = tmp_diff/non_zero_count
		for j in range(len(tmp_sample)) :
			if (imputed_dataset_reshape_df_scaled[sample][j] != 0.0) :
				imputed_dataset_reshape_df_scaled[sample][j] += add_pseudo
	imputed_dataset_reshape_df_scaled.index = imputed_dataset_reshape_df.index
	imputed_dataset_reshape_df_scaled.to_csv("imputed_dataset_from_DeepMicroGen_scaled.csv", mode = "w", index = True)
