import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)
import pickle
import numpy as np
import os

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import f1_score
#from sklearn.metrics import accuracy_score

import os
from tensorflow.python.client import device_lib
from collections import Counter
import time
f = open('data/glove.6B/glove.6B.300d.txt', 'r')
word_embedding = pickle.load(f)
f.close()
unknown_token = "UNKNOWN_TOKEN"

model_name = 'model-1-multigpu-1'
