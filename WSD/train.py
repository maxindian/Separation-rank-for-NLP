import tensorflow as tf
import pickle
from tensorflow.python.platform import gfile
from glove import *
import numpy as np
import sys
import math
import os
import random
import time
from six.moves import xrange
import models.idcnn
import util.vocabmapping
import time
import operator
from check_hyp_dic import *
import sys

word = 'hard'
num_classes = 3

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "config.ini", "Path to configuration file with hyper-parameters.")
flags.DEFINE_string("data_dir", "data/", "Path to main data directory.")
flags.DEFINE_string("checkpoint_dir", "data/checkpoints/", "Directory to store/restore checkpoints")

def main():
	hyper_params = check_get_hyper_param_dic(FLAGS)
	vocabmapping = util.vocabmapping.VocabMapping(word)
	vocab_size = vocabmapping.getSize()
	path = os.path.join(FLAGS.data_dir, "processed/")

	input_File=word+"_with_glove_vectors_100.npy"
	data = np.load(input_File)
	average_accuracy=0.0
	s=time.time()
	num_batches = len(data) // hyper_params["batch_size"]

	# 75/10/15 splir for train/ dev /test
	train_start_end_index = [0, int(hyper_params["train_frac"] * len(data))]
	test_start_end_index = [int((hyper_params["train_frac"]) * len(data)) + 1, len(data) - 1]
	
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		print("calling create_model")
		model = create_model(sess, hyper_params, vocab_size)
		writer = tf.summary.FileWriter("/tmp/tb_logs", sess.graph)
		
		tf.get_variable_scope().reuse_variables()
		print("Started Training...")
		step_time, loss = 0.0, 0.0
		previous_losses = []

		tot_steps = int(num_batches * hyper_params["max_epoch"])
		
		model.initData(data, train_start_end_index, test_start_end_index)
		
		X=[]
		Y=[]
		prediction_result=[]
		target_result=[]
		input_sentences_data=[]
		for step in xrange(1, tot_steps):
	
			start_time = time.time()
		
			inputs, targets, seq_lengths = model.getBatch()
	
			str_summary, step_loss, _ = model.step(sess, inputs, targets, seq_lengths, False)
		

			step_time += (time.time() - start_time) / hyper_params["steps_per_checkpoint"]
			loss += step_loss / hyper_params["steps_per_checkpoint"]
			
		
			if step % hyper_params["steps_per_checkpoint"] == 0:
				writer.add_summary(str_summary, step)
		
				print ("global step %d learning rate %.7f step-time %.2f loss %.4f"% (model.global_step.eval(), model.learning_rate.eval(),step_time, loss))
			# Decrease learning rate if no improvement was seen over last 3 times.

				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)
			
				step_time, loss, test_accuracy = 0.0, 0.0, 0.0
				print("Running test set")
			
				prediction_in_test=[]
				target_in_test=[]
				input_in_test=[]
				for test_step in xrange(len(model.test_data)):
					inputs, targets, seq_lengths = model.getBatch(True)
					str_summary, test_loss, _, accuracy,predictions,modified_target= model.step(sess, inputs, targets, seq_lengths, True)
					loss += test_loss
					test_accuracy += accuracy
					prediction_in_test.append(predictions)
					target_in_test.append(modified_target)
					input_in_test.append(inputs)
				prediction_result.append(prediction_in_test)
				target_result.append(target_in_test)
				input_sentences_data.append(input_in_test)
				normalized_test_loss, normalized_test_accuracy = loss / len(model.test_data), test_accuracy / (len(model.test_data)-2)
				checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "word_sense_disambiguation{0}.ckpt".format(normalized_test_accuracy))
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				writer.add_summary(str_summary, step)
				print("Avg Test Loss: {0}, Avg Test Accuracy: {1}".format(normalized_test_loss, normalized_test_accuracy))
				print("-------Step {0}/{1}------".format(step,tot_steps))
				loss = 0.0
				if test_step == len(model.test_data)-1:
					average_accuracy += normalized_test_accuracy
				sys.stdout.flush()

		input_sentences_data=np.asarray(input_sentences_data)
		a=np.asarray(prediction_result)
		b=np.asarray(target_result)
		np.save("out_predicted.npy", a)
		np.save("result_predicted.npy", b)
		
		pred = []
		target = []
		input_sentences_data_list = []
		for i in input_sentences_data[tot_steps//hyper_params["steps_per_checkpoint"]-2]:
			for j in i:
				input_sentences_data_list.append(j)
		for i in a[tot_steps//hyper_params["steps_per_checkpoint"]-2]:
			for j in i:
				pred.append(j)
		for i in b[tot_steps//hyper_params["steps_per_checkpoint"]-2]:
			for j in i:
				target.append(j)
		with open("util/"+word+"_index_2_word_map.txt", "rb") as handle:
			dic_sentences = pickle.loads(handle.read())
		with open("util/"+word+"_index_2_word_senses_map.txt","rb") as handle1:
			dic1_senses=pickle.loads(handle1.read())
		f2=open(word+'_mismatch_result.txt','w')
		for i ,j ,k in zip(input_sentences_data_list,pred,target):
			if j != k:
				sentence=[]
				for l in i:
					f2.write(str(dic_sentences[l])+' ')
					sentence.append(dic_sentences[l])

				f2.write(' MODEL_PREDICTION '+str(dic1_senses[j])+' GROUND_TRUTH '+str(dic1_senses[k])+' ')
			f2.write('\n')

		from sklearn.metrics import confusion_matrix
		from sklearn.metrics import f1_score
		f1 = f1_score(target, pred, average='macro')
		cnf_matrix = confusion_matrix(target,pred)

		print("Avg Test Accuracy: {0}, Avg Test F1 Score:{1}".format(normalized_test_accuracy, f1))

	end=time.time()
	print('time in training ',end-s)

def create_model(session, hyper_params, vocab_size):
	model = models.idcnn.WSDModel(vocab_size,
		hyper_params["hidden_size"],
		hyper_params["dropout"],
		hyper_params["num_layers"],
		hyper_params["grad_clip"],
		hyper_params["max_seq_length"],
		hyper_params["learning_rate"],
		hyper_params["lr_decay_factor"],
		hyper_params["batch_size"],
		word,
		num_classes)

	print("Created model with fresh parameters.")
	session.run(tf.global_variables_initializer())
	return model

if __name__ == '__main__':
	main()