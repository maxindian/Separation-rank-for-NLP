# Introduction
This code is used to verify the theoretical analysis from this paper:
Explainable Mechanism for Modeling Contextual Dependency in Neural Language Model  
This paper is submitted to EMNLP-2019 and under review. 
The Code is organized by six NLP tasks. They are Name Entity Recognition(NER), POS tagging, Word Sense Disambiguation (WSD), Coreference Resolution, Constituency parsing and sentiment analysis, respectively.
# Contents
## Environmet
Tensorflow 1.2+ torch 0.4+ python 3.5

## Name Entity Recognition 
Dataset: The CoNLL 2003[1]
We follow the language model augmented sequence taggers (TagLM)

## POS tagging
Dataset: Wall Street Journal of the Penn Treebank dataset[2]
Using the LSTM

## Word Sense Disambiguation
Dataset: the Senserval-2
We follow this model[3]

## Coreference Resolution 
Dataset: OntoNotes Release 5.0 benchmark[4]
We follow the work<An end to end coreference resolution>[5]   

## Sentiment analysis
Dataset: IMDB dataset [6]
We perform classification using a standard bidirectional LSTM with different hidden units and layers.

## The PLOT code


# Reference
[1]Erik F Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the conll-2003 shared task:
Language-independent named entity recognition. In Proceedings of the seventh conference on Natural
language learning at HLT-NAACL 2003-Volume 4, pages 142–147. Association for Computational Linguistics.
[2] Mitchell P. Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. 1993. Building a large annotated
corpus of English: the penn treebank.
[3] Mikael K°ageb¨ack and Hans Salomonsson. 2016. Word sense disambiguation using a bidirectional lstm. In
Proceedings of the 5th Workshop on Cognitive Aspects of the Lexicon (CogALex-V), pages 51–56.
[4]Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Olga Uryupina, and Yuchen Zhang. 2012. Conll-2012 shared task: Modeling multilingual unrestricted coreference in ontonotes. In Joint Conference on Emnlp & Conll-shared Task.
[5]Kenton Lee, Luheng He, Mike Lewis, and Luke Zettlemoyer. 2017. End-to-end neural coreference resolution.
[6]Andrew L Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher Potts. 2011.
Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association
for computational linguistics: Human language technologies-volume 1, pages 142–150. Association for Computational Linguistics.

