
import dynet_config
dynet_config.set_gpu()
import dynet as dy
import numpy as np

EPOCHS = 1
EMBEDDING_DIM = 300
ENCODE_DIM = 200

RNN_BUILDER = dy.LSTMBuilder


WORD2IDX = {}

"""
sample of an attention network from https://talbaumel.github.io/blog/attention/
"""
######################################################

# class SimpleRNNNetwork:
#     def __init__(self, rnn_num_of_layers, embeddings_size, state_size):
#         self.model = dy.Model()
#
#         # the embedding paramaters
#         self.embeddings = self.model.add_lookup_parameters((VOCAB_SIZE, embeddings_size))
#
#         # the rnn
#         self.RNN = RNN_BUILDER(rnn_num_of_layers, embeddings_size, state_size, self.model)
#
#         # project the rnn output to a vector of VOCAB_SIZE length
#         self.output_w = self.model.add_parameters((VOCAB_SIZE, state_size))
#         self.output_b = self.model.add_parameters((VOCAB_SIZE))
#
#     def _embed_string(self, sentence):
#         return [self.embeddings[word] for word in sentence]
#
#     def _run_rnn(self, init_state, input_vecs):
#         s = init_state
#
#         states = s.add_inputs(input_vecs)
#         rnn_outputs = [s.output() for s in states]
#         return rnn_outputs
#
#     def _get_probs(self, rnn_output):
#         output_w = dy.parameter(self.output_w)
#         output_b = dy.parameter(self.output_b)
#
#         probs = dy.softmax(output_w * rnn_output + output_b)
#         return probs
#
#     def __call__(self, input_string):
#         embedded_string = self._embed_string(input_string)
#         rnn_state = self.RNN.initial_state()
#         rnn_outputs = self._run_rnn(rnn_state, embedded_string)
#         return [self._get_probs(rnn_output) for rnn_output in rnn_outputs]
#
#     def get_loss(self, input_string, output_string):
#         input_string = self._add_eos(input_string)
#         output_string = self._add_eos(output_string)
#
#         dy.renew_cg()
#
#         probs = self(input_string)
#         loss = [-dy.log(dy.pick(p, output_char)) for p, output_char in zip(probs, output_string)]
#         loss = dy.esum(loss)
#         return loss
#
#     def _predict(self, probs):
#         probs = probs.value()
#         predicted_char = int2char[probs.index(max(probs))]
#         return predicted_char
#
#     def generate(self, input_string):
#         input_string = self._add_eos(input_string)
#
#         dy.renew_cg()
#
#         probs = self(input_string)
#         output_string = [self._predict(p) for p in probs]
#         output_string = ''.join(output_string)
#         return output_string.replace('<EOS>', '')
#
#
# class EncoderDecoderNetwork(SimpleRNNNetwork):
#     def __init__(self, enc_layers, dec_layers, enc_state_size, dec_state_size):
#         self.model = dy.Model()
#
#         # the embedding paramaters
#         self.word_embeddings = self.model.add_lookup_parameters((len(WORD2IDX) + 100, EMBEDDING_DIM))
#
#         # the rnns
#         self.ENC_RNN = RNN_BUILDER(enc_layers, EMBEDDING_DIM, enc_state_size, self.model)
#         self.DEC_RNN = RNN_BUILDER(dec_layers, enc_state_size, dec_state_size, self.model)
#
#         # project the rnn output to a vector of VOCAB_SIZE length
#         self.output_w = self.model.add_parameters((len(WORD2IDX), dec_state_size))
#         self.output_b = self.model.add_parameters((len(WORD2IDX)))
#
#     def _encode_string(self, embedded_string):
#         initial_state = self.ENC_RNN.initial_state()
#
#         # run_rnn returns all the hidden state of all the slices of the RNN
#         hidden_states = self._run_rnn(initial_state, embedded_string)
#
#         return hidden_states
#
#     def __call__(self, input_sentence):
#         embedded_sentence = self._embed_string(input_sentence)
#         # The encoded string is the hidden state of the last slice of the encoder
#         encoded_string = self._encode_string(embedded_sentence)[-1]
#
#         rnn_state = self.DEC_RNN.initial_state()
#
#         probs = []
#         for _ in range(len(input_sentence)):
#             rnn_state = rnn_state.add_input(encoded_string)
#             p = self._get_probs(rnn_state.output())
#             probs.append(p)
#         return probs
#
#
# class AttentionNetwork:
#     def __init__(self, enc_layers, dec_layers, embeddings_size, enc_state_size, dec_state_size):
#         EncoderDecoderNetwork.__init__(self, enc_layers, dec_layers, embeddings_size, enc_state_size, dec_state_size)
#
#         # attention weights
#         self.attention_w1 = self.model.add_parameters((enc_state_size, enc_state_size))
#         self.attention_w2 = self.model.add_parameters((enc_state_size, dec_state_size))
#         self.attention_v = self.model.add_parameters((1, enc_state_size))
#
#         self.enc_state_size = enc_state_size
#
#     def _attend(self, input_vectors, state):
#         w1 = dy.parameter(self.attention_w1)
#         w2 = dy.parameter(self.attention_w2)
#         v = dy.parameter(self.attention_v)
#         attention_weights = []
#
#         w2dt = w2 * state.h()[-1]
#         for input_vector in input_vectors:
#             attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
#             attention_weights.append(attention_weight)
#         attention_weights = dy.softmax(dy.concatenate(attention_weights))
#
#         output_vectors = dy.esum(
#             [vector * attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
#         return output_vectors
#
#     def __call__(self, input_string):
#         dy.renew_cg()
#
#         embedded_string = self._embed_string(input_string)
#         encoded_string = self._encode_string(embedded_string)
#
#         rnn_state = self.DEC_RNN.initial_state().add_input(dy.vecInput(self.enc_state_size))
#
#         probs = []
#         for _ in range(len(input_string)):
#             attended_encoding = self._attend(encoded_string, rnn_state)
#             rnn_state = rnn_state.add_input(attended_encoding)
#             p = self._get_probs(rnn_state.output())
#             probs.append(p)
#         return probs

######################################################


class AttentionNetwork:
    def __init__(self):
        self.model = dy.Model()

        self.word_embeddings = self.model.add_lookup_parameters((len(WORD2IDX) + 100, EMBEDDING_DIM))

        self.w_embed = self.model.add_parameters((ENCODE_DIM, EMBEDDING_DIM))
        self.b_embed = self.model.add_parameters(ENCODE_DIM)

    def _attend(self, input_vectors):
        w = dy.parameter(self.w_embed)
        b = dy.parameter(self.b_embed)
        output_vectors = []

        for input_vector in input_vectors:
            output_vectors = dy.rectify(w * input_vector + b)
            output_vectors.append(output_vectors)

        return output_vectors

    def __call__(self, sent1, sent2):
        dy.renew_cg()


def process_sentence(sentence, from_train=False):
    """
    Get a sentence as string and convert it to indexes.
    :param sentence: string
    :param from_train: is sentence from train set
    :return: sentence indexes
    """
    sentence = sentence.split()

    # remove dot
    sentence[-1] = sentence[-1][:-1]

    if from_train:
        global WORD2IDX

        for word in sentence:
            if word not in WORD2IDX:
                WORD2IDX[word] = len(WORD2IDX)

    return sentence_to_indexs(sentence)


def sentence_to_indexs(sentence):
    """
    Take a list of words(parse sentence) and convert it to indexes.
    :param sentence: list of words(parse sentence)
    :return: words indexes, OOV words are mapped by hash
    """
    return np.array([WORD2IDX[word] if word in WORD2IDX
                     else len(WORD2IDX) + (hash(word) % 100) - 1 for word in sentence])


with open('snli_1.0/snli_1.0_train.txt') as train_file, open('snli_1.0/snli_1.0_dev.txt') as dev_file,\
        open('snli_1.0/snli_1.0_test.txt') as test_file:
    train_content = [np.take(np.array(line.split('\t')), [0, 3, 4]) for line in (train_file.readlines())]
    dev_content = [np.take(np.array(line.split('\t')), [0, 3, 4]) for line in (dev_file.readlines())]
    test_content = [np.take(np.array(line.split('\t')), [0, 3, 4]) for line in (test_file.readlines())]


# get data to form of ([sentence1, sentence2], gold_label)
# sentence{1, 2} are indexes for words
train_content = [([process_sentence(sample[1], from_train=True), process_sentence(sample[2], from_train=True)],
                  sample[0]) for sample in train_content if sample[0] != '-'][:50]
dev_content = [([process_sentence(sample[1]), process_sentence(sample[2])], sample[0])
               for sample in dev_content if sample[0] != '-']
test_content = [([process_sentence(sample[1]), process_sentence(sample[2])], sample[0])
                for sample in test_content if sample[0] != '-']

# for e in range(EPOCHS):


