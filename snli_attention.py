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
RUN_DEV = 500
LR = 0.01
NUM_UNK = 100

EMBEDDING_SIZE = 300
NUM_LAYERS = 2
STATE_SIZE = 100
ATTENTION_SIZE = 100
LABELS = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
START = '<s>'
END = '</s>'


class Attention:
    def __init__(self, glove_vectors, word2glove, embedding_size, tag_size, lr):
        # (self, vocab_size, embedding_size, num_layers, state_size, attention_size, tag_size):
        self.model = dy.Model()
        self.trainer = dy.AdagradTrainer(self.model, lr)

        # Embedding vectors
        self.embedding = glove_vectors
        self.w2idx = word2glove
        # self.input_lookup = self.model.add_lookup_parameters((vocab_size + NUM_UNK, embedding_size))

        # Feed forward NN encoder F
        self.F_W1 = self.model.add_parameters((embedding_size, embedding_size))
        self.F_W2 = self.model.add_parameters((embedding_size, embedding_size))
        # self.enc_fwd_lstm = dy.LSTMBuilder(num_layers, embedding_size, state_size, self.model)
        # self.enc_bwd_lstm = dy.LSTMBuilder(num_layers, embedding_size, state_size, self.model)

        # Feed Forward NN aligner G
        self.G_W1 = self.model.add_parameters((embedding_size, 2 * embedding_size))
        self.G_W2 = self.model.add_parameters((embedding_size, embedding_size))
        # self.dec_lstm = dy.LSTMBuilder(num_layers, state_size * 2 + embedding_size, state_size, self.model)

        # Feed Forward NN decoder H
        self.H_W1 = self.model.add_parameters((embedding_size, 2 * embedding_size))
        self.H_W2 = self.model.add_parameters((tag_size, embedding_size))

        # self.attention_w1 = self.model.add_parameters((attention_size, state_size * 2))
        # self.attention_w2 = self.model.add_parameters((attention_size, state_size * num_layers * 2))
        # self.attention_v = self.model.add_parameters((1, attention_size))
        # self.decoder_w = self.model.add_parameters((tag_size, state_size))
        # self.output_lookup = self.model.add_lookup_parameters((vocab_size, embedding_size))

        self.activation = dy.rectify

    def embed_sentence(self, sentence):
        sentence = sentence[:-1]
        sentence = [START] + sentence.split() + [END]
        """for w in sentence:
            idx = self.w2idx[w]
            e = self.embedding[idx]
            e = dy.inputTensor(e)"""
        return [dy.inputTensor(self.embedding[self.w2idx.get(w, self.w2idx['<UNK/>'])]) for w in sentence]
        """sentence = [word2idx_dict.get(w, 0) for w in sentence]
        return [self.input_lookup[word] for word in sentence]"""

    def encode_sentence(self, embed_sent):
        def F(word):
            f_w1 = self.activation(self.F_W1 * dy.dropout(word, 0.85))
            f_w2 = self.activation(self.F_W2 * dy.dropout(f_w1, 0.85))
            return f_w2
        return dy.concatenate_cols([F(w) for w in embed_sent])
        """fwd_vectors = self.run_lstm(self.enc_fwd_lstm.initial_state(), sentence)
        bwd_vectors = reversed(self.run_lstm(self.enc_bwd_lstm.initial_state(), reversed(sentence)))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors"""

    def attend1(self, encode_sent1, encode_sent2):
        sent1_matrix = dy.transpose(encode_sent1) * encode_sent2
        sent2_matrix = dy.transpose(sent1_matrix)
        sent1_weights, sent2_weights = dy.softmax(sent1_matrix), dy.softmax(sent2_matrix)
        return sent1_weights * dy.transpose(encode_sent2), sent2_weights * dy.transpose(encode_sent1)

    def __call__(self, sent1, sent2):
        dy.renew_cg()
        embed_sent1, embed_sent2 = self.embed_sentence(sent1), self.embed_sentence(sent2)
        encoded_sent1, encoded_sent2 = self.encode_sentence(embed_sent1), self.encode_sentence(embed_sent2)
        attend_sent1, attend_sent2 = self.attend1(encoded_sent1, encoded_sent2)
        v_sent1, v_sent2 = self.compare(encoded_sent1, attend_sent1), self.compare(encoded_sent2, attend_sent2)
        prediction = self.aggregate(v_sent1, v_sent2)
        return prediction

    def compare(self, sent, v_sent_other):
        def G(vec1, vec2):
            g_w1 = self.activation(self.G_W1 * dy.dropout(dy.concatenate([vec1, vec2]), 0.85))
            g_w2 = self.activation(self.G_W2 * dy.dropout(g_w1, 0.85))
            return dy.transpose(g_w2)
        return dy.sum_dim(dy.concatenate([G(w, v_soft_align) for w, v_soft_align in zip(dy.transpose(sent), v_sent_other)]), d=[0])

    def aggregate(self, v_sent1, v_sent2):
        h_w1 = self.activation(self.H_W1 * dy.concatenate([v_sent1, v_sent2]))
        h_w2 = self.activation(self.H_W2 * h_w1)
        return dy.softmax(h_w2)

    def train(self, train_set, dev_set):
        train_accuracy = []
        dev_accuracy = []
        for epoch in range(1, EPOCHS + 1):
            print('EPOCH #' + str(epoch) + ':')
            loss_list = []
            correct = 0.0
            # need to shuffle data
            train_set = shuffle(train_set)
            for item in train_set[:1000]:
                sent1, sent2, label = item['sentence1'], item['sentence2'], LABELS[item['gold_label']]
                prediction_vec = self(sent1, sent2)
                prediction = np.argmax(prediction_vec.npvalue())
                loss = dy.pickneglogsoftmax(prediction_vec, label)
                loss_list.append(loss)
                loss.backward()
                self.trainer.update()
                correct += 1 if prediction == label else 0
            accuracy = str(100 * (correct / len(train_set)))
            train_accuracy.append(accuracy)
            print('Accuracy on train set: ' + accuracy + '%')

            correct = 0.0
            for item in dev_set[:500]:
                sent1, sent2, label = item['sentence1'], item['sentence2'], LABELS[item['gold_label']]
                prediction_vec = self(sent1, sent2)
                prediction = np.argmax(prediction_vec.npvalue())
                correct += 1 if prediction == label else 0
            accuracy = str(100 * (correct / len(dev_set)))
            dev_accuracy.append(accuracy)
            print('Accuracy on train set: ' + accuracy + '%')
            print('\n')
            """loss = self.get_loss([text, hypothesis, label], word2idx_dict, glove_embedding)
                loss_value = loss.value()
                loss.backward()
                trainer.update()
                if epoch % 20 == 0:
                    print(loss_value)
                    print(self.generate(sentence, word2idx_dict, idx2word_dict))"""
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
    word_dict = {w: i for i, w in enumerate(set().union(*[words1, words2, [START, END]]))}
    inv_word_dict = {i: w for w, i in word_dict.items()}
    return word_dict, inv_word_dict


if __name__ == '__main__':
    """new_glove_file()
    exit()"""
    start_time = time.time()
    train_data, dev_data = load_data(DATA_DIR + TRAIN), load_data(DATA_DIR + DEV)
    curr_time = time.time()
    print('Loaded train and dev data in ' + str(curr_time - start_time) + ' seconds\n')
    start_time = curr_time
    # word2idx, idx2word = create_word2idx(train_data)
    glove_matrix, w2glove = load_glove(GLOVE)
    curr_time = time.time()
    print('Created word2idx and idx2word dictionaries in ' + str(curr_time - start_time) + ' seconds\n')
    # attention_model = Attention(len(word2idx), EMBEDDING_SIZE, NUM_LAYERS, STATE_SIZE, ATTENTION_SIZE, tag_size=3)
    attention_model = Attention(glove_matrix, w2glove, EMBEDDING_SIZE, len(LABELS), LR)
    train_acc, dev_acc = attention_model.train(train_data, dev_data)


"""def attend(self, input_mat, state, w1dt):
        w2 = dy.parameter(self.attention_w2)
        v = dy.parameter(self.attention_v)

        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = w2 * dy.concatenate(list(state.s()))
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(v * self.activation(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        # context: (encoder_state)
        context = input_mat * att_weights
        return context

    def decode(self, vectors, output, word2idx_dict):
        text_vec, hypothesis_vec = vectors

        output = START + output.split() + END
        output = [word2idx_dict[w] for w in output]

        w = dy.parameter(self.decoder_w)
        w1 = dy.parameter(self.attention_w1)
        input_mat = dy.concatenate_cols(vectors)
        w1dt = None

        last_output_embeddings = self.output_lookup[word2idx_dict[END]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))
        loss = []

        for char in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = w * s.output()
            probs = dy.softmax(out_vector)
            last_output_embeddings = self.output_lookup[char]
            loss.append(-dy.log(dy.pick(probs, char)))
        loss = dy.esum(loss)
        return loss"""

"""def generate(self, in_seq, word2idx_dict, idx2word_dict):
        embedded = self.embed_sentence(in_seq, word2idx_dict)
        encoded = self.encode_sentence(embedded)

        w = dy.parameter(self.decoder_w)
        w1 = dy.parameter(self.attention_w1)
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = self.output_lookup[word2idx_dict[END]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

        out = ''
        count_EOS = 0
        for i in range(len(in_seq) * 2):
            if count_EOS == 2: break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = w * s.output()
            probs = dy.softmax(out_vector).vec_value()
            next_char = probs.index(max(probs))
            last_output_embeddings = self.output_lookup[next_char]
            if idx2word_dict[next_char] == END:
                count_EOS += 1
                continue

            out += idx2word_dict[next_char]
        return out

    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors

    def get_loss(self, input_data, word2idx_dict, glove_embedding):
        dy.renew_cg()
        sent1, sent2, label = input_data
        embedded_text, embedded_hypothesis = self.embed_sentence(sent1, word2idx_dict, glove_embedding), self.embed_sentence(sent2, word2idx_dict, glove_embedding)
        encoded_text, encoded_hypothesis = self.encode_sentence(embedded_text), self.encode_sentence(embedded_hypothesis)
        a_m = self.attend1(encoded_text, encoded_hypothesis)
        return self.decode([encoded_text, encoded_hypothesis], label, word2idx_dict)"""