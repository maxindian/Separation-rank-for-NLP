import nltk
import pickle
word = "hard"
number_of_senses = 3
def createVocab(max_vocab_size):
	print("Creating vocab mapping ...")
	dic = {}
	f=open("wsd_datasets/"+word+"/"+word+"_sentences.txt","r")
	f1=open("wsd_datasets/"+word+"/"+word+"_senses.txt","r")
	for line in f.readlines():
		tokens = line.lower().split()
		for t in tokens:
			if t not in dic:
				dic[t] = 1