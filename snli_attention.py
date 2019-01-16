import json
import dynet as dy
import time
import numpy as np
import pickle
from sklearn.utils import shuffle

DATA_DIR = 'snli_1.0'
TRAIN = '/snli_1.0_train.jsonl'
DEV = '/snli_1.0_dev.jsonl'
TEST = '/snli_1.0_test.jsonl'
GLOVE = 'glove.42B.300d.txt'

EPOCHS = 10
LR = 0.08

EMBEDDING_DIM = 300
HIDDEN_DIM = 300
ATTENTION_SIZE = 100
LABELS = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
EOS = '<EOS/>'
UNK = '<UNK/>'


class Attention:
    def __init__(self, glove_vectors, word2glove, embedding_dim, hidden_dim, tag_size, lr):
        self.model = dy.Model()
        self.trainer = dy.AdagradTrainer(self.model, lr)

        # Embedding vectors
        self.embedding = glove_vectors
        self.w2idx = word2glove

        # Feed forward NN encoder F
        self.F_W1 = self.model.add_parameters((embedding_dim, hidden_dim))
        self.F_W2 = self.model.add_parameters((hidden_dim, hidden_dim))

        # Feed Forward NN aligner G
        self.G_W1 = self.model.add_parameters((2 * hidden_dim, hidden_dim))
        self.G_W2 = self.model.add_parameters((hidden_dim, hidden_dim))

        # Feed Forward NN decoder H
        self.H_W1 = self.model.add_parameters((hidden_dim, 2 * hidden_dim))
        self.H_W2 = self.model.add_parameters((tag_size, hidden_dim))

        self.activation = dy.rectify

    def embed_sentence(self, sentence):
        sentence = sentence[:-1].split() + [EOS]
        return dy.inputTensor(np.array([self.embedding[self.w2idx.get(w, self.w2idx[UNK])] for w in sentence]))

    def encode_sentence(self, embed_sent):
        f_w1 = self.activation(dy.dropout(embed_sent, 0.15) * self.F_W1)
        f_w2 = self.activation(dy.dropout(f_w1, 0.15) * self.F_W2)
        return f_w2

    def attend(self, encode_sent1, encode_sent2):
        sent1_matrix = encode_sent1 * dy.transpose(encode_sent2)
        sent2_matrix = dy.transpose(sent1_matrix)
        sent1_weights, sent2_weights = dy.softmax(sent1_matrix), dy.softmax(sent2_matrix)
        return sent1_weights * encode_sent2, sent2_weights * encode_sent1

    def predict(self, sent1, sent2):
        dy.renew_cg()
        embed_sent1, embed_sent2 = self.embed_sentence(sent1), self.embed_sentence(sent2)
        encoded_sent1, encoded_sent2 = self.encode_sentence(embed_sent1), self.encode_sentence(embed_sent2)
        attend_sent1, attend_sent2 = self.attend(encoded_sent1, encoded_sent2)
        v_sent1, v_sent2 = self.compare(encoded_sent1, attend_sent1), self.compare(encoded_sent2, attend_sent2)
        prediction = self.aggregate(v_sent1, v_sent2)
        return prediction

    def compare(self, sent, v_sent_other):
        vec = dy.concatenate_cols([sent, v_sent_other])
        g_w1 = self.activation(dy.dropout(vec, 0.15) * self.G_W1)
        g_w2 = self.activation(dy.dropout(g_w1, 0.15) * self.G_W2)
        return dy.sum_dim(g_w2, d=[0])

    def aggregate(self, v_sent1, v_sent2):
        h_w1 = self.activation(self.H_W1 * dy.concatenate([v_sent1, v_sent2]))
        h_w2 = self.activation(self.H_W2 * h_w1)
        return h_w2

    def train(self, train_set, dev_set):
        train_accuracy = []
        dev_accuracy = []
        for epoch in range(1, EPOCHS + 1):
            print('EPOCH #' + str(epoch) + ':')
            correct = 0.0
            train_set = shuffle(train_set)
            for i, item in enumerate(train_set[:1000]):
                sent1, sent2, label = item['sentence1'], item['sentence2'], LABELS[item['gold_label']]
                prediction_vec = self.predict(sent1, sent2)
                prediction = np.argmax(dy.softmax(prediction_vec).npvalue())
                loss = dy.pickneglogsoftmax(prediction_vec, label)
                loss.backward()
                self.trainer.update()
                correct += 1 if prediction == label else 0
            accuracy = str(100 * (correct / 1000))
            train_accuracy.append(accuracy)
            print('Accuracy on train set: ' + accuracy + '%')

            correct = 0.0
            for item in dev_set[:500]:
                sent1, sent2, label = item['sentence1'], item['sentence2'], LABELS[item['gold_label']]
                prediction_vec = self.predict(sent1, sent2)
                prediction = np.argmax(prediction_vec.npvalue())
                correct += 1 if prediction == label else 0
            accuracy = str(100 * (correct / 500))
            dev_accuracy.append(accuracy)
            print('Accuracy on dev set: ' + accuracy + '%')
            print('\n')
        return train_accuracy, dev_accuracy


def load_data(f_name):
    lines = [json.loads(line) for line in open(f_name)]
    return [line for line in lines if line['gold_label'] != '-']


def load_glove(f_name):
    print("Loading Glove Model")
    glove_embeds = pickle.load(open('new_glove', 'rb'))
    return np.array([vec for vec in glove_embeds.values()]), {v: i for i, v in enumerate(glove_embeds.keys())}
    """with open(f_name, 'r', encoding='utf8') as file:
        model = {}
        for line in file:
            fields = line.split()
            word = fields[0]
            embedding = np.array([float(val) for val in fields[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        return model"""


def new_glove_file():
    train_data, dev_data, test_data = load_data(DATA_DIR + TRAIN), load_data(DATA_DIR + DEV), load_data(DATA_DIR + TEST)
    set1, set2, set3 = create_word2idx(train_data)[0], create_word2idx(dev_data)[0], create_word2idx(test_data)[0]
    all_keys = set().union(*[set1.keys(), set2.keys(), set3.keys()])
    print('starting creating glove')
    with open(GLOVE, 'r', encoding='utf8') as file:
        model = {}
        for line in file:
            fields = line.split()
            word = fields[0]
            embedding = np.array([float(val) for val in fields[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        new_model = {}
        not_in_glove = ['<UNK/>', START, END]
        for key in all_keys:
            if key in model:
                new_model[key] = model[key]
            else:
                not_in_glove.append(key)
        vec_len = len(model['a'])
        for key in not_in_glove:
            new_model[key] = np.random.rand(vec_len)
            print(key)
        print('dumping glove')
        pickle.dump(new_model, open('new_glove', 'wb'))


def create_word2idx(data):
    words1 = [w for item in data for w in item['sentence1'].split()]
    words2 = [w for item in data for w in item['sentence2'].split()]
    word_dict = {w: i for i, w in enumerate(set().union(*[words1, words2, [EOS, UNK]]))}
    inv_word_dict = {i: w for w, i in word_dict.items()}
    return word_dict, inv_word_dict


if __name__ == '__main__':
    start_time = time.time()
    train_data, dev_data = load_data(DATA_DIR + TRAIN), load_data(DATA_DIR + DEV)
    curr_time = time.time()
    print('Loaded train and dev data in ' + str(curr_time - start_time) + ' seconds\n')
    start_time = curr_time
    glove_matrix, w2glove = load_glove(GLOVE)
    curr_time = time.time()
    print('Created word2idx and idx2word dictionaries in ' + str(curr_time - start_time) + ' seconds\n')
    attention_model = Attention(glove_matrix, w2glove, EMBEDDING_DIM, HIDDEN_DIM, len(LABELS), LR)
    train_acc, dev_acc = attention_model.train(train_data, dev_data)
