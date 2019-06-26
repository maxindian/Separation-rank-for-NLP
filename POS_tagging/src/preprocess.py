import glob
from random import shuffle

MAX_LENGTH = 100

prefixes_common = ['anti', 'ante', 'an', 'auto', 'bi', 'circum', 'contra', 'counter', 'de', 'dis', 'di', 'exo', 'extra', 'extro', 'fore', 'hemi', 'hyper', 'hypo', 'il', 'im' 'in', 'ir', 'inter', 'intra', 'macro', 'mal', 'micro', 'mis', 'mono', 'multi', 'non', 'post', 'pre', 're', 'semi', 'sub', 'super', 'tri', 'ultra', 'uni', 'un']
suffixes_common = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist', 'ity', 'ty', 'ment', 'ness', 'ship', 'sion', 'tion', 'ate', 'en', 'ify', 'fy', 'ize', 'ise', 'able', 'ible', 'al', 'esque', 'ful', 'ical', 'ic', 'ious', 'ous', 'ish', 'ive', 'less', 'y']

prefixes_very_common = ['anti', 'auto', 'bi', 'circum', 'counter', 'dis', 'hyper', 'il', 'im' 'in', 'ir', 'inter', 'intra', 'macro', 'mal', 'micro', 'mis', 'mono', 'multi', 'non', 'post', 'pre', 're', 'semi', 'sub', 'super', 'ultra', 'uni', 'un']
suffixes_very_common = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist', 'ity', 'ty', 'ment', 'ness', 'ship', 'sion', 'tion', 'ate', 'en', 'ify', 'fy', 'able', 'ible', 'al', 'ful', 'ical', 'ic', 'ious', 'ous', 'ish', 'ive', 'less', 'y']

prefixes_exhaustive = ['anti', 'ante', 'an', 'a', 'ab', 'ad', 'ac', 'as', 'auto', 'ben', 'bi', 'circum', 'com', 'con', 'co', 'contra', 'counter', 'de', 'dis', 'di', 'eu', 'exo', 'extra', 'extro', 'ecto', 'ex', 'fore', 'hemi', 'hyper', 'hypo', 'il', 'im' 'in', 'ir', 'inter', 'intra', 'macro', 'mal', 'micro', 'mis', 'mono', 'multi', 'non', 'ob', 'oc', 'op', 'o', 'omni', 'peri', 'poly', 'post', 'pre', 'pro', 'quad', 're', 'semi', 'sub', 'super', 'supra', 'sym', 'syn', 'trans', 'tri', 'ultra', 'uni', 'un']
suffixes_exhaustive = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist', 'ity', 'ty', 'ment', 'ness', 'ship', 'sion', 'tion', 'ate', 'en', 'ify', 'fy', 'ize', 'ise', 'able', 'ible', 'al', 'esque', 'ful', 'ical', 'ic', 'ious', 'ous', 'ish', 'ive', 'less', 'y']

class PreprocessData:
    prefixes = prefixes_exhaustive
    suffixes = suffixes_exhaustive

    # prefixes = prefixes_exhaustive
    # suffixes = suffixes_exhaustive

    def __init__(self, dataset_type='wsj'):
        self.vocabulary = {}
        self.pos_tags = {}
        self.prefix_orthographic = {}
        self.suffix_orthographic = {}
        self.dataset_type = dataset_type

        # Some special prefix or suffix values
        self.prefix_orthographic['none'] = 0
        self.prefix_orthographic['capital'] = 1
        self.prefix_orthographic['startnum'] = 2

        self.suffix_orthographic['none'] = 0
        self.suffix_orthographic['hyphen'] = 1

    def get_prefix_feature_id(self, token, mode):
        if token[0].isdigit():
            return self.prefix_orthographic['startnum']
        if token[0].isupper():
            return self.prefix_orthographic['capital']
        for prefix in self.prefixes:
            # print 'Checking prefix ', prefix
            if token.startswith(prefix):
                return self.get_orthographic_id(prefix, self.prefix_orthographic, mode)
        return self.prefix_orthographic['none']

    def get_suffix_feature_id(self, token, mode):
        if "-" in token:
            return self.suffix_orthographic['hyphen']
        for suffix in self.suffixes:
            if token.endswith(suffix):
                return self.get_orthographic_id(suffix, self.suffix_orthographic, mode)
        return self.prefix_orthographic['none']

    ## Get standard split for WSJ
    def get_standard_split(self, files):
        if self.dataset_type == 'wsj':
            train_files = []
            val_files = []
            test_files = []
            for file_ in files:
                partition = int(file_.split('/')[-2])
                if partition >= 0 and partition <= 18:
                    train_files.append(file_)
                elif partition <= 21:
                    val_files.append(file_)
                else:
                    test_files.append(file_)
            return train_files, val_files, test_files
        else:
            raise Exception('Standard Split not Implemented for ' + self.dataset_type)

    @staticmethod
    def isFeasibleStartingCharacter(c):
        unfeasibleChars = '[]@\n'
        return not (c in unfeasibleChars)

    ## OOV words represented by len(vocab)
    def get_oov_id(self, dic):
        return len(dic)

    def get_pad_id(self, dic):
        return len(self.vocabulary) + 1

    # Adds a token to orthographic map during training mode
    # During test/validation, returns -1 if it is not seen during training
    def get_orthographic_id(self, pos, dic, mode):
        if pos not in dic:
            # Irrespective of test or validation or training phase, we add the feature to the map and assign a new id
            dic[pos] = len(dic)
            # if mode == 'train':
            #     dic[pos] = len(dic)
            # else:
            #     return self.prefix_orthographic['none'] # or self.suffix_orthographic['none']
        return dic[pos]

    ## get id of given token(pos) from dictionary dic.
    ## if not in dic, extend the dic if in train mode
    ## else use representation for unknown token
    # pos:word
    def get_id(self, pos, dic, mode):
        if pos not in dic:
            if mode == 'train':
                dic[pos] = len(dic)
            else:
                return self.get_oov_id(dic)
        return dic[pos]

    ## Process single file to get raw data matrix
    def processSingleFile(self, inFileName, mode):
        matrix = []
        row = []
        word_count = 0
        with open(inFileName) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    pass
                else:
                    tokens = line.split()
                    for token in tokens:
                        ## ==== indicates start of new example
                        if token[0] == '=':
                            if row:
                                matrix.append(row)
                            word_count = 0
                            row = []
                            break
                        elif PreprocessData.isFeasibleStartingCharacter(token[0]):
                            wordPosPair = token.split('/')
                            # The MAX_LENGTH check ensures that the training vocabularly
                            # only includes in those words that are finally a part of the
                            # training instance (not the clipped off portion)
                            if len(wordPosPair) == 2 and word_count < MAX_LENGTH:
                                word_count += 1
                                ## get ids for word
                                feature = self.get_id(wordPosPair[0], self.vocabulary, mode)

                                # get ids for prefix and suffix features
                                prefix_feature = self.get_prefix_feature_id(wordPosPair[0], mode)
                                suffix_feature = self.get_suffix_feature_id(wordPosPair[0], mode)

                                # get id for pos tag. Instead of passing input mode
                                # we pass train as the mode so that we can include all pos tags
                                row.append((feature, self.get_id(wordPosPair[1],
                                                                 self.pos_tags, 'train'), prefix_feature, suffix_feature))
        if row:
            matrix.append(row)
        return matrix

    # get all data files in given subdirectories of given directory
    def preProcessDirectory(self, inDirectoryName, subDirNames=['*']):
        if not (subDirNames):
            files = glob.glob(inDirectoryName + '/*.pos')
        else:
            files = [glob.glob(inDirectoryName + '/' + subDirName + '/*.pos')
                     for subDirName in subDirNames]
            files = set().union(*files)
        return list(files)

    ## Get basic data matrix with (possibly) variable sized senteces, without padding
    def get_raw_data(self, files, mode):
        matrix = []
        for f in files:
            matrix.extend(self.processSingleFile(f, mode))
        return matrix

    def split_data(self, data, fraction):
        split_index = int(fraction * len(data))
        left_split = data[:split_index]
        right_split = data[split_index:]
        if not (left_split):
            raise Exception('Fraction too small')
        if not (right_split):
            raise Exception('Fraction too big')
        return left_split, right_split

    ## Get rid of sentences greater than max_size
    ## and pad the remaining if less than max_size
    def get_processed_data(self, mat, max_size):
        X = []
        y = []
        P = []
        S = []
        original_len = len(mat)
        mat = filter(lambda x: len(x) <= max_size, mat)
        mat = list(mat)
        no_removed = original_len - len(mat)
        print("no_removed={}".format(no_removed))
        for row in mat:
            X_row = [tup[0] for tup in row]
            y_row = [tup[1] for tup in row]
            P_row = [tup[2] for tup in row]
            S_row = [tup[3] for tup in row]
            ## padded words represented by len(vocab) + 1
            X_row = X_row + [self.get_pad_id(self.vocabulary)] * (max_size - len(X_row))
            ## Padded pos tags represented by -1
            y_row = y_row + [-1] * (max_size - len(y_row))
            P_row = P_row + [self.prefix_orthographic['none']] * (max_size - len(P_row))
            S_row = S_row + [self.suffix_orthographic['none']] * (max_size - len(S_row))
            X.append(X_row)
            y.append(y_row)
            P.append(P_row)
            S.append(S_row)
        return X, y, P, S, self.vocabulary
