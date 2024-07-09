import numpy as np
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
import re

def load_baseline_topic_words(path):
    topic_words = {}
    relations = {}
    l = -1
    with open(path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if not line: continue
            data = line.split(':')
            if len(data) < 2:
                l += 1
                topic_words[l] = {}
                relations[l] = {}
                continue
            pre = re.findall(r"\d+", data[0])
            kw = eval(data[1])

            tid = int(pre[0])
            topic_words[l][tid] = kw
            if l > 0: 
                child = [int(d) for d in pre[1:]]
                relations[l-1][tid] = child
    return topic_words, relations

def load_baseline_topic_words_v2(path):
    tid_dict = {}
    tid_l = 0
    topic_words = {}
    relations = {}
    l = -1
    with open(path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if not line: continue
            data = line.split(':')
            if len(data) < 2:
                l += 1
                tid_dict[l] = {}
                tid_l = 0
                topic_words[l] = {}
                relations[l] = {}
                continue
            pre = re.findall(r"\d+", data[0])
            kw = eval(data[1])
            tid = int(pre[0])

            if l > 0: 
                child = [tid_dict[l-1][int(d)] for d in pre[1:] if int(d) in tid_dict[l-1]]
                if len(child) == 0: continue
                relations[l-1][tid_l] = child

            topic_words[l][tid_l] = kw
            tid_dict[l][tid] = tid_l
            tid_l += 1
    return topic_words, relations

def compute_overlap(level1, level2):
    sum_overlap_score = 0.0
    for N in [5, 10, 15]:
        word_idx1 = level1[:N]
        word_idx2 = level2[:N]
        total = min(len(word_idx1), len(word_idx2))
        if total == 0: continue
        c = 0
        for n in word_idx1:
            if n in word_idx2:
                c += 1
        sum_overlap_score += c / total
    return sum_overlap_score / 3

def compute_cl_diversity(level1, level2):
    cl_div = 0.
    for N in [5, 10, 15]:
        word_idx1 = level1[:N]
        word_idx2 = level2[:N]
        topic_words = [word_idx1, word_idx2]
        vocab = set(sum(topic_words,[]))
        total = sum(topic_words,[])
        cl_div += len(vocab) / len(total)
    return cl_div / 3.0

def evaluate_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words,[]))
    total = sum(topic_words,[])
    if not total: return 1.0
    return len(vocab) / len(total)

def compute_topic_specialization(topic_word, corpus_topic):
    topics_vec = topic_word
    if topics_vec.shape[0] > 0:
        for i in range(topics_vec.shape[0]):
            topics_vec[i] = topics_vec[i] / np.linalg.norm(topics_vec[i])
        topics_spec = 1 - topics_vec.dot(corpus_topic)
        depth_spec = np.mean(topics_spec)
        return depth_spec
    else:
        return 0

def compute_clnpmi(level1, level2, doc_word):
    sum_coherence_score = 0.0
    c = 0
    for N in [5,10,15]:
        word_idx1 = level1[:N]
        word_idx2 = level2[:N]
        
        sum_score = 0.0
        set1 = set(word_idx1)
        set2 = set(word_idx2)
        inter = set1.intersection(set2)
        word_idx1 = list(set1.difference(inter))
        word_idx2 = list(set2.difference(inter))

        for n in range(len(word_idx1)):
            flag_n = doc_word[:, word_idx1[n]] > 0
            p_n = np.sum(flag_n) / len(doc_word)
            for l in range(len(word_idx2)):
                flag_l = doc_word[:, word_idx2[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_nl == len(doc_word):
                    sum_score += 1
                elif p_n * p_l * p_nl > 0:
                    p_l = p_l / len(doc_word)
                    p_nl = p_nl / len(doc_word)
                    p_nl += 1e-10
                    score = np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
                    if score > 0: sum_score += score
                c += 1
        if c > 0:
            sum_score /= c
        else:
            sum_score = 0
        sum_coherence_score += sum_score
    return sum_coherence_score / 3


def compute_coherence(doc_word, topic_word_idx, topk):
    # print('computing coherence ...')    
    topic_size = np.shape(topic_word_idx)[0]
    doc_size = np.shape(doc_word)[0]
    # find top words'index of each topic
    topic_list = topic_word_idx

    # compute coherence of each topic
    sum_coherence_score = 0.0
    for i in range(topic_size):
        word_array = topic_list[i]
        sum_score = 0.0
        N = min(topk, len(word_array))
        if N < 2: continue
        for n in range(N):
            flag_n = doc_word[:, word_array[n]] > 0
            p_n = np.sum(flag_n) / doc_size
            for l in range(n + 1, N):
                flag_l = doc_word[:, word_array[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_n * p_l * p_nl > 0:
                    p_l = p_l / doc_size
                    p_nl = p_nl / doc_size
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        sum_coherence_score += sum_score * (2 / (N * N - N))
    sum_coherence_score = sum_coherence_score / (topic_size + 1e-7)
    return sum_coherence_score

# from gensim
def evaluate_coherence(topic_words, texts, vocab):
    coherence = {}
    methods = ["c_v", "c_npmi", "c_uci", "u_mass"]
    for method in methods:
        coherence[method] = CoherenceModel(topics=topic_words, texts=texts, dictionary=vocab, coherence=method).get_coherence()
    return coherence

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

def plot_loss(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc="best")