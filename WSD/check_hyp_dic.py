import os
import util.hyperparams as hyperparams
from glove import *
import configparser
import time 
def check_get_hyper_param_dic(FLAGS):

	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	serializer = hyperparams.HyperParameterHandler(FLAGS.checkpoint_dir)
	hyper_params = read_config_file(FLAGS)
	if serializer.checkExists():
		if serializer.checkChanged(hyper_params):
			if not hyper_params["use_config_file_if_checkpoint_exists"]:
				hyper_params = serializer.getParams()
				print("Restoring hyper params from previous checkpoint...")
			else:
				new_checkpoint_dir = "{0}_hidden_size_{1}_numlayers_{2}_dropout_{3}".format(
				int(time.time()),
				hyper_params["hidden_size"],
				hyper_params["num_layers"],
				hyper_params["dropout"])
				new_checkpoint_dir = os.path.join(FLAGS.checkpoint_dir,
					new_checkpoint_dir)
				os.makedirs(new_checkpoint_dir)
				FLAGS.checkpoint_dir = new_checkpoint_dir
				serializer = hyperparams.HyperParameterHandler(FLAGS.checkpoint_dir)
				serializer.saveParams(hyper_params)
		else:
			print("No hyper parameter changed detected, using old checkpoint...")
	else:
		serializer.saveParams(hyper_params)
		print("No hyper params detected at checkpoint... reading config file")
	return hyper_params
def read_config_file(FLAGS):
	'''
	Reads in config file, returns dictionary of network params
	'''
	config = configparser.ConfigParser()
	config.read(FLAGS.config_file)
	dic = {}
	wsd_section = "WSD_network_params"
	general_section = "general"
	dic["num_layers"] = config.getint(wsd_section, "num_layers")
	dic["hidden_size"] = config.getint(wsd_section, "hidden_size")
	dic["dropout"] = config.getfloat(wsd_section, "dropout")
	dic["batch_size"] = config.getint(wsd_section, "batch_size")
	dic["train_frac"] = config.getfloat(wsd_section, "train_frac")
	dic["learning_rate"] = config.getfloat(wsd_section, "learning_rate")
	dic["lr_decay_factor"] = config.getfloat(wsd_section, "lr_decay_factor")
	dic["grad_clip"] = config.getint(wsd_section, "grad_clip")
	dic["use_config_file_if_checkpoint_exists"] = config.getboolean(general_section,
		"use_config_file_if_checkpoint_exists")
	dic["max_epoch"] = config.getint(wsd_section, "max_epoch")
	dic ["max_vocab_size"] = config.getint(wsd_section, "max_vocab_size")
	dic["max_seq_length"] = config.getint(general_section,
		"max_seq_length")
	dic["steps_per_checkpoint"] = config.getint(general_section,
		"steps_per_checkpoint")
	return dic
