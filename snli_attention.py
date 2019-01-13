import json
import dynet as dy
import time

DATA_DIR = 'snli_1.0'
TRAIN = '/snli_1.0_train.jsonl'
DEV = '/snli_1.0_dev.jsonl'
TEST = '/snli_1.0_test.jsonl'

EPOCHS = 600
NUM_UNK = 100

EMBEDDING_SIZE = 300
NUM_LAYERS = 2
STATE_SIZE = 100
ATTENTION_SIZE = 100
START = '<s>'
END = '</s>'


class Attention:
    def __init__(self, vocab_size, embedding_size, num_layers, state_size, attention_size, tag_size):
        self.model = dy.Model()

        self.enc_fwd_lstm = dy.LSTMBuilder(num_layers, embedding_size, state_size, self.model)
        self.enc_bwd_lstm = dy.LSTMBuilder(num_layers, embedding_size, state_size, self.model)

        self.dec_lstm = dy.LSTMBuilder(num_layers, state_size * 2 + embedding_size, state_size, self.model)

        self.input_lookup = self.model.add_lookup_parameters((vocab_size, embedding_size))
        self.attention_w1 = self.model.add_parameters((attention_size, state_size * 2))
        self.attention_w2 = self.model.add_parameters((attention_size, state_size * num_layers * 2))
        self.attention_v = self.model.add_parameters((1, attention_size))
        self.decoder_w = self.model.add_parameters((vocab_size, state_size))
        self.output_lookup = self.model.add_lookup_parameters((vocab_size, embedding_size))

        self.activation = dy.rectify

    def embed_sentence(self, sentence, word2idx_dict):
        sentence = START + sentence.split() + END
        sentence = [word2idx_dict[w] for w in sentence]
        return [self.input_lookup[word] for word in sentence]

    def encode_sentence(self, sentence):
        fwd_vectors = self.run_lstm(self.enc_fwd_lstm.initial_state(), sentence)
        bwd_vectors = reversed(self.run_lstm(self.enc_bwd_lstm.initial_state(), reversed(sentence)))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors

    def attend(self, input_mat, state, w1dt):
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
        return loss

    def generate(self, in_seq, word2idx_dict, idx2word_dict):
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

    def get_loss(self, input_sentence, output_sentence, word2idx_dict):
        dy.renew_cg()
        embedded = self.embed_sentence(input_sentence, word2idx_dict)
        encoded = self.encode_sentence(embedded)
        return self.decode(encoded, output_sentence, word2idx_dict)

    def train(self, train_set, dev_set, word2idx_dict, idx2word_dict):
        trainer = dy.AdagradTrainer(self.model)
        for epoch in range(1, EPOCHS + 1):
            print('EPOCH #: ' + str(epoch))
            loss = self.get_loss(sentence, sentence, word2idx_dict)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            if epoch % 20 == 0:
                print(loss_value)
                print(self.generate(sentence, word2idx_dict, idx2word_dict))


def load_data(f_name):
    return [json.loads(line) for line in open(f_name)]


def create_word2idx(data):
    words1 = [w for item in data for w in item['sentence1'].split()]
    words2 = [w for item in data for w in item['sentence2'].split()]
    word_dict = {i: w for i, w in enumerate(set().union(*[words1, words2, [START, END]]))}
    inv_word_dict = {w: i for i, w in word_dict.items()}
    return word_dict, inv_word_dict


if __name__ == '__main__':
    start_time = time.time()
    train_data, dev_data = load_data(DATA_DIR + TRAIN), load_data(DATA_DIR + DEV)
    curr_time = time.time()
    print('Loaded train and dev data in ' + str(curr_time - start_time) + ' seconds\n')
    start_time = curr_time
    word2idx, idx2word = create_word2idx(train_data)
    curr_time = time.time()
    print('Created word2idx and idx2word dictionaries in ' + str(curr_time - start_time) + 'seconds\n')
    attention_model = Attention(len(word2idx), EMBEDDING_SIZE, NUM_LAYERS, STATE_SIZE, ATTENTION_SIZE, tag_size=3)
    attention_model.train(train_data, dev_data, word2idx, idx2word)
