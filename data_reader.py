from itertools import count
import os
from typing import Counter
import numpy as np
import pickle
from numpy.core.fromnumeric import shape
from scipy.sparse import csr_matrix

# from collections import Counter
import gensim
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import STOPWORDS
from collections import defaultdict
from sklearn.preprocessing import normalize as sknormalize
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print(" [*] save {}".format(path))

def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        print(" [*] load {}".format(path))
        return obj

class TextReader():
    def __init__(self, data_path, emb_size=50):
        print("Dataset preparing....")
        self.base_path = data_path
        self.vocab_path = os.path.join(self.base_path, "processed/vocab.pkl")

        # load or rebuild vocab
        try:
            self._load()
        except:
            self._rebuild_vocab(frequence=50)

        self.idx2word = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.emb_size = emb_size
        print(f"Load {self.vocab_size} words..")

    # read original text
    # [sample] label \t text
    def _read_text(self, file_path):
        labels = []
        texts = []
        with open(file_path, encoding='utf-8') as f:
            docs = open(file_path).read().split('\n')
            for n_i, line in enumerate(docs):
                if line == '':
                    continue
                label, text = line.strip().split('\t', 1)
                text = list(gensim.utils.tokenize(text, lower=True))
                texts.append(text)
                labels.append(label)

            return labels, texts

    def _build_vocab(self, frequence):
        self.label_dict = {}
        try:
            with open(os.path.join(self.base_path, 'raw/labels.txt'), 'r') as f:
                f = f.read()
                l = f.split('\n')
                for id, label in enumerate(l):
                    self.label_dict[label.lower()] = id
        except:
            pass
        # read train / valid / test split
        self._read_data()

        all_text = self.train_text + self.valid_text + self.test_text

        # build dictionary
        dictionary = gensim.corpora.Dictionary(all_text)
        print("Before", len(dictionary))
        dictionary = self._filter_(dictionary, frequence)
        print("After", len(dictionary))
        self.vocab = dictionary
        save_pkl(self.vocab_path, self.vocab)

    # preprocess original text
    def _filter_(self, dictionary, frequence):
        # remove stopwords
        dictionary.filter_tokens(list(map(dictionary.token2id.get, STOPWORDS)))

        # remove words with < 3 characters
        len_1_words = list(filter(lambda w: len(w) < 2, dictionary.values()))
        dictionary.filter_tokens(
            list(map(dictionary.token2id.get, len_1_words)))

        # remove low-frequency words
        dictionary.filter_extremes(no_below=frequence)
        dictionary.compactify()

        return dictionary

    def _save_text(self, text, path):
        with open(path, 'w') as f:
            for document in text:
                f.write('0\t')
                f.write(' '.join(document))
                f.write('\n')
    
    def _convert_bow_to_text(self):
        train_bow = self.get_bow("train")
        valid_bow = self.get_bow("valid")
        test_bow  = self.get_bow("test")

        train_text = self._bow_to_text(train_bow, self.vocab.id2token)
        valid_text = self._bow_to_text(valid_bow, self.vocab.id2token)
        test_text = self._bow_to_text(test_bow, self.vocab.id2token)

        # save text
        self._save_text(train_text,
                        os.path.join(self.base_path, "raw/train_bow.txt"))
        self._save_text(valid_text,
                        os.path.join(self.base_path, "raw/valid_bow.txt"))
        self._save_text(test_text, os.path.join(self.base_path,
                                                "raw/test_bow.txt"))

    def _bow_to_text(self, bow, id2token):
        text = []
        for document in bow:
            d = []
            for word in document:
                d.extend([word[0]] * word[1])
            t = list(map(id2token.get, d))
            text.append(t)
        return text

    def _read_from_bow(self, data_path):
        file_data = open(data_path).read().strip().split('\n')
        documents = []
        for d in file_data:
            document = []
            if d.strip() == "":
                continue
            for word_fre in d.strip().split(' '):
                token, fre = word_fre.split(':')
                if 'rcv1' in data_path:
                    token = int(token)
                else:
                    token = int(token) - 1
                fre = int(fre)
                document.append((token, fre))
            # documents.append([ tuple(map(int, word_fre.split(':'))) for word_fre in d.split(' ')])
            documents.append(document)
        return documents

    def _build_from_curpus(self, vocab_path):
        id2token = {}
        with open(vocab_path, 'r') as f:
            words = f.read().strip().split('\n')
            for id, word in enumerate(words):
                id2token[id] = word.split()[0]

        train_bow = self._read_from_bow(
            vocab_path.replace('data.vocab', 'train.feat'))
        test_bow = self._read_from_bow(
            vocab_path.replace('data.vocab', 'test.feat'))

        self.vocab = gensim.corpora.Dictionary.from_corpus(
            train_bow + test_bow, id2token)
        save_pkl(self.vocab_path, self.vocab)

        # build from bow
        trunc_count = int(len(train_bow) / 20)
        valid_bow = train_bow[-trunc_count:]
        train_bow = train_bow[:len(train_bow) - trunc_count]

        train_text = self._bow_to_text(train_bow, id2token)
        valid_text = self._bow_to_text(valid_bow, id2token)
        test_text = self._bow_to_text(test_bow, id2token)

        # save text
        self._save_text(train_text,
                        os.path.join(self.base_path, "raw/train.txt"))
        self._save_text(valid_text,
                        os.path.join(self.base_path, "raw/valid.txt"))
        self._save_text(test_text, os.path.join(self.base_path,
                                                "raw/test.txt"))

    def _rebuild_vocab(self, frequence=50):
        train_path = os.path.join(self.base_path, "raw/train.txt")
        valid_path = os.path.join(self.base_path, "raw/valid.txt")
        test_path = os.path.join(self.base_path, "raw/test.txt")

        if os.path.exists(os.path.join(self.base_path, "raw/data.vocab")):
            self._build_from_curpus(
                os.path.join(self.base_path, "raw/data.vocab"))
        else:
            self._build_vocab(frequence)
        self.train_data, self.train_label, self.train_text = self._file_to_data(
            train_path)
        self.valid_data, self.valid_label, self.valid_text = self._file_to_data(
            valid_path)
        self.test_data, self.test_label, self.test_text = self._file_to_data(
            test_path)

    def _read_data(self):
        train_path = os.path.join(self.base_path, "raw/train.txt")
        valid_path = os.path.join(self.base_path, "raw/valid.txt")
        test_path = os.path.join(self.base_path, "raw/test.txt")

        train_label, self.train_text = self._read_text(train_path)
        valid_label, self.valid_text = self._read_text(valid_path)
        test_label, self.test_text = self._read_text(test_path)

    def _file_to_data(self, file_path):
        labels, texts = self._read_text(file_path)

        data = []
        m_labels = []
        for label, text in zip(labels, texts):
            # remove documents with less than 3 tokens
            if len(self.vocab.doc2bow(text)) < 3:
                continue

            word = list(map(self.vocab.token2id.get, text))
            word = np.array(list(filter(lambda x: x is not None, word)),
                            dtype=object)

            if label.isdigit():
                m_labels.append(int(label))
            elif ':' in label:
                l = label.split(':')[0]
                m_labels.append(self.label_dict[l.lower()])
            else:
                m_labels.append(self.label_dict[label.lower()])
            data.append(word)

        lens = list(map(len, data))
        print(" [*] load {} docs, avg len: {}, max len: {}".format(
            len(data), np.mean(lens), np.max(lens)))

        data = np.array(data, dtype=object)
        save_pkl(
            file_path.replace('/raw/',
                              '/processed/').replace('.txt', '_data.pkl'),
            data)

        m_labels = np.array(m_labels, dtype=object)
        if m_labels.max() == self.get_n_classes():
            m_labels = m_labels - 1
        save_pkl(
            file_path.replace('/raw/',
                              '/processed/').replace('.txt', '_label.pkl'),
            m_labels)
        return data, m_labels, texts

    # load data from original text
    def _load(self):
        # read original text
        self._read_data()

        processed_train_path = os.path.join(self.base_path,
                                            'processed/train_data.pkl')
        processed_valid_path = os.path.join(self.base_path,
                                            'processed/valid_data.pkl')
        processed_test_path = os.path.join(self.base_path,
                                           'processed/test_data.pkl')

        # load processed vocab & data & label
        self.vocab = load_pkl(self.vocab_path)

        self.train_data = load_pkl(processed_train_path)
        self.train_label = load_pkl(
            processed_train_path.replace('_data.pkl',
                                         '_label.pkl')).astype(np.int)

        self.valid_data = load_pkl(processed_valid_path)
        self.valid_label = load_pkl(
            processed_valid_path.replace('_data.pkl',
                                         '_label.pkl')).astype(np.int)

        self.test_data = load_pkl(processed_test_path)
        self.test_label = load_pkl(
            processed_test_path.replace('_data.pkl',
                                        '_label.pkl')).astype(np.int)

    def get_n_classes(self):
        if hasattr(self, "n_classes"):
            return self.n_classes
        else:
            with open(self.base_path + '/raw/labels.txt', 'r') as f:
                f = f.read()
                l = f.split('\n')
                self.n_classes = len(l)
                return self.n_classes

    # get train/valid/test/all text and labels
    def get_sequence(self, data_type):
        if data_type == "train":
            return (self.train_data, self.train_label, self.train_text)

        elif data_type == "valid":
            return (self.valid_data, self.valid_label, self.valid_text)

        elif data_type == "test":
            return (self.test_data, self.test_label, self.test_text)

        elif data_type == "train+valid":
            data = np.concatenate([self.train_data, self.valid_data])
            label = np.concatenate([self.train_label, self.valid_label])
            text = np.concatenate([self.train_text, self.valid_text])
            return (data, label, text)

        elif data_type == "all":
            data = np.concatenate(
                [self.train_data, self.valid_data, self.test_data])
            label = np.concatenate(
                [self.train_label, self.valid_label, self.test_label])
            text = np.concatenate(
                [self.train_text, self.valid_text, self.test_text])
            return (data, label, text)
        else:
            raise Exception(" [!] Unkown data type : {}".format(data_type))

    def generator_sequence(self, data_type="train", batch_size=32, rand=True):
        raw_data, raw_label, raw_text = self.get_sequence(data_type)

        count = 0
        while True:
            if not rand:
                beg = (count * batch_size) % raw_data.shape[0]
                end = ((count + 1) * batch_size) % raw_data.shape[0]
                if beg > end:
                    beg -= raw_data.shape[0]

                idx = np.arange(beg, end)
            else:
                idx = np.random.randint(0, len(raw_data), batch_size)

            data = raw_data[idx]
            label = raw_label[idx]
            text = raw_text[idx]

            yield [data, label, text]

            count += 1

    # data in matrix format
    # 4 modes：['onehot', 'count', 'tfidf', 'freq']
    def get_matrix(self, data_type="train", mode='onehot', normalize=False):
        raw_data, raw_label, raw_text = self.get_sequence(data_type)

        matrix_path = os.path.join(
            self.base_path,
            f'processed/matrix_{data_type}_{mode}_{normalize}.pkl')

        if os.path.exists(matrix_path):
            x = load_pkl(matrix_path)
            return [x, raw_label, raw_text]

        x = np.zeros((len(raw_data), self.vocab_size))
        for i, seq in enumerate(raw_data):
            counter = defaultdict(int)
            for j in seq: counter[j] += 1

            # 填充矩阵元素
            for j, c in list(counter.items()):
                if mode == 'count':
                    x[i][j] = c
                elif mode == 'fred':
                    x[i][j] = c / len(seq)
                elif mode == 'onehot':
                    x[i][j] = 1
                elif mode == 'tfidf':
                    # tf = 1 + np.log(c)
                    tf = np.log(1 + c)
                    idf = np.log(1 + self.vocab.num_docs /
                                 (1 + self.vocab.dfs[j]))
                    x[i][j] = tf * idf
                else:
                    raise ValueError('Unknown vectorization mode:', mode)

        if normalize:
            # x = x / x.sum(axis=1, keepdims=True)
            x = sknormalize(x, norm='l1', axis=1)
            x = x.astype(np.float32)

        return [x, raw_label, raw_text]

    def generator_matrix(self,
                         data_type="train",
                         batch_size=32,
                         rand=True,
                         mode='onehot',
                         normalize=False):
        raw_data, raw_label, raw_text = self.get_sequence(data_type)

        count = 0
        while True:
            if not rand:
                beg = (count * batch_size) % raw_data.shape[0]
                end = ((count + 1) * batch_size) % raw_data.shape[0]
                if beg > end:
                    beg -= raw_data.shape[0]

                idx = np.arange(beg, end)
            else:
                idx = np.random.randint(0, len(raw_data), batch_size)

            data = raw_data[idx]
            label = raw_label[idx]
            text = raw_text[idx]

            x = np.zeros((len(data), self.vocab_size))
            for i, seq in enumerate(data):
                counter = defaultdict(int)
                for j in seq:
                    counter[j] += 1

                for j, c in list(counter.items()):
                    if mode == 'count':
                        x[i][j] = c
                    elif mode == 'fred':
                        x[i][j] = c / len(seq)
                    elif mode == 'onehot':
                        x[i][j] = 1
                    elif mode == 'tfidf':
                        tf = 1 + np.log(c)
                        idf = np.log(1 + self.vocab.num_docs /
                                     (1 + self.vocab.dfs[j]))

                        x[i][j] = tf * idf
                    else:
                        raise ValueError('Unknown vectorization mode:', mode)

            if normalize:
                # x = x / x.sum(axis=1, keepdims=True)
                x = sknormalize(x, norm='l1', axis=1)
                x = x.astype(np.float32)

            yield [x, label, text]

            count += 1

    # word_id sequence -> bow
    def get_bow(self, data_type="train"):
        raw_data, raw_label, raw_text = self.get_sequence(data_type)

        bow = []
        for item in raw_data:
            bow.append(list(Counter(item).items()))
        return bow

    # save bigdata in format of sparse matrix
    def get_sparse_matrix(self,
                          data_type="train",
                          mode='onehot',
                          normalize=False):
        raw_data, raw_label, raw_text = self.get_sequence(data_type)

        sparse_matrix_path = os.path.join(
            self.base_path,
            f'processed/sparse_matrix_{data_type}_{mode}_{normalize}.pkl')

        if os.path.exists(sparse_matrix_path):
            x = load_pkl(sparse_matrix_path)
            return [x, raw_label, raw_text]

        row = []
        col = []
        data = []
        for i, seq in enumerate(raw_data):
            counter = defaultdict(int)
            for j in seq:
                counter[j] += 1
            for j, c in list(counter.items()):
                row.append(i)
                col.append(j)
                if mode == 'count':
                    data.append(c)
                elif mode == 'fred':
                    data.append(c / len(seq))
                elif mode == 'onehot':
                    data.append(1)
                elif mode == 'tfidf':
                    tf = 1 + np.log(c)
                    idf = np.log(1 + self.vocab.num_docs /
                                 (1 + self.vocab.dfs[j]))
                    data.append(tf * idf)
                else:
                    raise ValueError('Unknown vectorization mode:', mode)

        data, row, col = np.array(data), np.array(row), np.array(col)

        x = csr_matrix(
            (data, (row, col)),
            shape=(len(raw_data), self.vocab_size)).astype(np.float32)

        if normalize:
            # x = x / x.sum(axis=1, keepdims=True)
            x = sknormalize(x, norm='l1', axis=1)
            x = x.astype(np.float32)

        save_pkl(sparse_matrix_path, x)

        return [x, raw_label, raw_text]
