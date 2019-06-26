# BiLSTM-POS-Tagging
Part of Speech tagging using Bidirectional Long Short Term Memory (LSTM). Trying to add orthographic features (like prefix, suffix etc) to the BiLSTM to different places (like input or output) and comparing the accuracy obtained.

======================= Instructions to run =========================  
cd src/  
python -u pos_bilstm.py <path_to_data_dir> <path_to_train_directory> standard <train|test> <INPUT|OUTPUT|NONE> <ONE_HOT|EMBEDDED|INT_VALUES>  
Default place is NONE (Baseline) and default type is ONE_HOT (type doesn't matter if type is NONE)  
  
Example 1: python -u pos_bilstm.py ../data/wsj/ ../train1 standard train INPUT ONE_HOT   (Input with ONE_HOT)  
Example 2: python -u pos_bilstm.py ../data/wsj/ ../train2 standard test OUTPUT EMBEDDED  (Output with EMBEDDED)  
Example 3: python -u pos_bilstm.py ../data/wsj/ ../train3 standard train NONE            (Baseline)  
