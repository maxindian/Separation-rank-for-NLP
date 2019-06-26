import subprocess

if __name__ == '__main__':

    # Tsize=[600, 650, 700]
    for i in range(3, 7):
        state_size = i * 50
        a = subprocess.check_call(['python', '-u', 'pos_bilstm_rnn.py', '../data/wsj/', '../em100_{}_f1_2_rnn'.format(state_size), 'standard', 'train', str(state_size), 'NONE'])
        print(a)

