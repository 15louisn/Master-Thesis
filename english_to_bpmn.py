import json

import tensorflow as tf

from sklearn.model_selection import train_test_split

import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors, fasttext
from gensim.test.utils import datapath, get_tmpfile

import nltk
from nltk.translate.bleu_score import sentence_bleu as bleu
from nltk.translate.bleu_score import SmoothingFunction

import unicodedata
import re
import numpy as np
import os
import io
import time
import warnings
import statistics

max_length_translation = 9999

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # adding space between punctuation and characters
    w = re.sub(r"([?.!,])", r" \1 ", w)

    # removing multiple spaces
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)

    w = w.strip()

    # adding the start and end tokens to the sentence
    w = '<start> ' + w + ' <end>'
    return w

def preprocess_graph(w):
    # Remove header
    w = w.split("\n", 1)[1]

    # Remove unsuitable sections
    w = re.sub(r"<bpmndi:BPMNDiagram((.|\n)*?)bpmndi:BPMNDiagram>", r"", w)
    w = re.sub(r"<bpmn:collaboration((.|\n)*?)bpmn:collaboration>", r"", w)

    # Renumber IDs
    keywords = re.findall(r"(?<=id=\").*?_", w)
    keywords += re.findall(r"(?<=ef=\").*?_", w)
    keywords = set(keywords)

    for key in keywords:
        IDs = re.findall(r"(?<=\""+ key + r").*?\"", w)
        i = 0
        for id in IDs:
            w = w.replace(id, str(i) + " ")
            i += 1

    # w = unicode_to_ascii(w.lower().strip())
    w = unicode_to_ascii(w.lower().strip())

    # adding space between punctuation and characters
    w = re.sub(r"([?.!,])", r" \1 ", w)

    # remove \n
    w = re.sub(r"(\n)", " ", w)

    w = re.sub(r"(>)", r"> ", w)

    w = re.sub(r"(<)", r" <", w)

    w = re.sub(r"(_)", r"", w)

    # removing multiple spaces
    w = re.sub(r'[" "]+', " ", w)

    w = w.strip()

    # adding the start and end tokens to the sentence
    w = '<start> ' + w + ' <end>'
    return w

def preprocess_graph_max(w):
    w = w.split("\n", 1)[1]
    w = re.sub(r"<bpmndi:BPMNDiagram((.|\n)*?)bpmndi:BPMNDiagram>", r"", w)
    w = re.sub(r"<bpmn:collaboration((.|\n)*?)bpmn:collaboration>", r"", w)

    keywords = re.findall(r"(?<=id=\").*?", w)
    keywords += re.findall(r"(?<=ef=\").*?", w)
    keywords = set(keywords)

    for key in keywords:
        IDs = re.findall(r"(?<=\""+ key + r").*?\"", w)
        i = 0
        for id in IDs:
            w = w.replace(id, str(i) + " ")
            i += 1

    w = re.sub(r"(_)", r"", w)
    w = re.sub(r"id=", r"",w)
    w = re.sub(r"(?<= ).*?ef=", r"",w)
    w = re.sub(r"</.*?>", r"",w)

    # w = unicode_to_ascii(w.lower().strip())
    w = unicode_to_ascii(w.lower().strip())

    # adding space between punctuation and characters
    w = re.sub(r"([?.!,])", r" \1 ", w)

    # remove \n
    w = re.sub(r"(\n)", " ", w)

    w = re.sub(r"(>)", r"> ", w)

    w = re.sub(r"(<)", r" <", w)

    # removing multiple spaces
    w = re.sub(r'[" "]+', " ", w)

    w = w.strip()

    # adding the start and end tokens to the sentence
    w = '<start> ' + w + ' <end>'
    return w


def check_files(dir_path):
    i = 0
    model_path = os.path.join(dir_path, 'Models')
    for file in [f for f in os.listdir(model_path) if f.endswith('.bpmn')]:
        i += 1

    print("Number of BPMN files: ", i, "\n\n")


def create_dataset(dir_path, num_examples):
    i = 0
    bpmn_path = os.path.join(dir_path, 'Models')
    text_path = os.path.join(dir_path, 'Texts')
    bpmns = []
    texts = []

    test = False
    for file in [f for f in os.listdir(bpmn_path) if f.endswith('.bpmn')]:
        if test:
            print(file)
            test = False

        with open(os.path.join(bpmn_path, file),'r') as of:
            if test:
                print(preprocess_graph_2(of.read()))
            bpmns.append(preprocess_graph_2(of.read()))

        (filename, ext) = os.path.splitext(file)


        with open(os.path.join(text_path, filename + '.txt'), encoding='latin-1') as of:
            if test:
                print(preprocess_sentence(of.read()))
                test = False
            texts.append(preprocess_sentence(of.read()))

        i += 1

        if num_examples is not None and i >= num_examples:
            break

    print("Number of BPMN files: ", i,"\n\n")

    # texts_length = []
    # bpmns_length = []
    # for x, y in zip(texts, bpmns):
    #     texts_length.append(len(x))
    #     bpmns_length.append(len(y))
    #
    # print("Text average size: ", sum(texts_length)/i)
    # print("BPMNs average size: ", sum(bpmns_length)/i)
    # print("BPMNs min size: ", min(bpmns_length))
    # print("BPMNs max size: ", max(bpmns_length))

    return texts, bpmns

# Tokenize all the sentences contained in lang and pad them.
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    # tokenize the sentences
    tensor = lang_tokenizer.texts_to_sequences(lang)

    # pad the tokenized sentences with '0' so that they always have the same length
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    # return the tokenized sentences as well as the tokenizer which contains the mapping between the
    # words and their related number
    return tensor, lang_tokenizer

# Pad (non-tokenized) sentences with empty strings so that they always have the same length.
def pad_sentences(lang):
    max_size = len(max(lang, key=len))

    for sentence in lang:
        while(len(sentence) < max_size):
            sentence.append('')

    return lang

def load_dataset(paths, num_examples=None):
    # load the preprocessed sentences
    input_lang, target_lang = [], []

    for p in paths:
        input_lang_temp, target_lang_temp = create_dataset(p, num_examples)
        input_lang += input_lang_temp
        target_lang += target_lang_temp

    # tokenize the target sentences and stored them in a tensor
    target_tensor, target_lang_tokenizer = tokenize(target_lang)

    # split the sentences into list of words
    input_lang = [[w for w in s.split()] for s in input_lang]
    target_lang = [[w for w in s.split()] for s in target_lang]

    # pad the splitted sentences to always have the same length
    input_lang = pad_sentences(input_lang)
    target_lang = pad_sentences(target_lang)

    return input_lang, target_lang, target_tensor, target_lang_tokenizer

def ft_custom_create(sentences, embed_file):
    if os.path.exists(embed_file):
        print('Previous adequate custom embeddings found, loading model from file:')
        print(embed_file)
        return KeyedVectors.load(embed_file)

    model = fasttext.FastText(sentences, size=emb_dims, window=10, min_count=0, workers=4).wv

    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    model.save(embed_file)
    return model

def ft_pretrained_create(sentences, embed_file):
    dp = datapath(resources_folder+"data/cc.en.300.bin")
    model = fasttext.load_facebook_vectors(dp)

    return model

# Return the embeddings of the 'sos' symbol, with the same dimensions as 'model'.
def get_sos(model):
    return tf.random.stateless_normal(
        (model.vector_size,), mean=0.0, stddev=0.00001, seed=(sos_seed, 1))

# Return the embeddings of the 'eos' symbol, with the same dimensions as 'model'.
def get_eos(model):
    return tf.random.stateless_normal(
        (model.vector_size,), mean=0.0, stddev=0.00001, seed=(eos_seed, 1))

# Return the list of embeddings of a sentence.
def embed(sentence, model):
    embedded = []
    for w in sentence:
        # w has been added for padding, its embeddings are all 0's
        if w == '':
            embedded.append(tf.zeros(model.vector_size))

        # w is the sos
        elif w == '<start>':
            embedded.append(get_sos(model))

        # w is the eos
        elif w == '<end>':
            embedded.append(get_eos(model))

        # w is not known by the embedder
        elif not w in model.wv:
            embedded.append(tf.zeros(model.vector_size))

        # normal case where w is a word known by the embedder
        else:
            embedded.append(model.wv[w])

    return embedded

# Return the embeddings of a list of sentences.
def embed_all_sentences(lang, model):
    return tf.convert_to_tensor([embed(s, model) for s in lang])


class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        # LSTM layer
        self.rnn = tf.keras.layers.LSTM(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

    # Returns the output and the two states of the LSTM layer.
    def call(self, x, hidden):
        output, hidden_state, cell_state = self.rnn(x, initial_state=hidden)

        return output, (hidden_state, cell_state)

    # Returns the initial states of the two LSTM layers, i.e. tensors with only 0's.
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units

        # attention layer
        if attention is True:
            self.attention = BahdanauAttention(self.dec_units)

        # LSTM layer
        self.rnn = tf.keras.layers.LSTM(self.dec_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

        # FC layer for generating the output word tokens
        self.fc = tf.keras.layers.Dense(vocab_target_size)

    # Returns the scores of the word tokens and the two states of the LSTM layer.
    def call(self, x, hidden, enc_output=None):
        if attention:
            context_vector = self.attention(hidden[0], enc_output)

            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, hidden_state, cell_state = self.rnn(x, initial_state=hidden)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, (hidden_state, cell_state)



# Returns cross-entropy loss while applying the padding mask.
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# Computes the loss over a batch, updates the models and returns the loss
@tf.function
def train_step(input_lang, target_lang, target_tensor, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(input_lang, enc_hidden)

        dec_hidden = enc_hidden

        # Teacher forcing - feeding the target as the next input
        for t in range(target_lang.shape[1] - 1):
            dec_input = tf.expand_dims(target_lang[:, t], axis=1)

            predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(target_tensor[:, t+1], predictions)

    batch_loss = (loss / int(target_lang.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# Transform a numpy array of bites string into a list of lists of words (= list of sentences)
def to_list_strings(a):
    sentences = np.empty(a.shape, dtype=object)

    for (x, y), w in np.ndenumerate(a):
        sentences[x, y] = w.decode('utf-8')

    return sentences

# Returns the translated sentence given the embeddings of the input sequence.
def translate_embed(embeddings):
    # initial state of the encoder
    hidden = tf.zeros((1, units)), tf.zeros((1, units))

    # pass through the encoder
    enc_output, enc_hidden = encoder(tf.expand_dims(embeddings, axis=0), hidden)

    dec_hidden = enc_hidden
    l = 0
    word_input_embed = get_sos(embedder_text)
    translation = []

    # loop for generating the translation
    while l < max_length_translation:
        # pass through the decoder
        dec_output, dec_hidden = decoder(
            tf.reshape(word_input_embed, (1, 1, word_input_embed.shape[0])), dec_hidden, enc_output)

        print("BAN\n\n\n\n",dec_output)
        dec_output = tf.reshape(dec_output, (dec_output.shape[1]))

        # get the most probable token
        token_output = tf.math.argmax(dec_output).numpy()

        # get the word associated to this token
        if token_output == 0:
            word_output = '<unk>'

        else:
            word_output = target_lang_tokenizer.index_word[token_output]

        # stop the translating process if the decoder has outputed an eos
        if word_output == '<end>':
            return translation

        translation.append(word_output)

        # get the embeddings of the outputed word, which will be the next inputs of the decoder
        word_input_embed = tf.convert_to_tensor(embed([word_output], embedder_text)[0])

        l += 1

    return translation

# Translate a sentence using the trained model.
def translate_text_to_graph(sentence):
    # # preprocess sentence and split it into words
    # sentence = preprocess_sentence(sentence).split()

    # get the embeddings
    embeddings = tf.convert_to_tensor(embed(sentence, embedder_text))

    # translate
    return translate_embed(embeddings)

if __name__ == "__main__":
    # Source: https://github.com/setzer22/alignment_model_text
    old_dataset_path = "/Users/louisnelissen/Documents/github/alignment_model_text/datasets/OriginalDataset"
    new_dataset_path = "/Users/louisnelissen/Documents/github/alignment_model_text/datasets/NewDataset"

    paths = [old_dataset_path, new_dataset_path]

    num_examples = 2

    # General constants
    drive = False                               # Set to 'True' to store the results on Google Drive
    restore = False                             # Set to 'True' if the model must be restored
    train = True                                # Set to 'True' if the model must be trained

    resources_folder = "/Users/louisnelissen/Documents/Resources/TFE/"
    results_folder = resources_folder + 'results/'                 # Where to store the results
    checkpoint_dir = resources_folder + 'training_checkpoints/'    # Where to store the weights
    embeddings_dir = resources_folder + 'custom_embeddings/'       # Where to store the custom embeddings model

    # Dataset
    num_examples = None                         # Num of sentences taken from the dataset ('None' for all)
    test_size = 0.1                             # Ratio of sentences used for evaluation
    split_seed = 42                             # Seed used for splitting the dataset

    # Embeddings
    emb_dims = 100                              # Number of dimensions used for the custom embeddings
    sos_seed = 42                               # Seed used for generating the 'sos' symbol's embeddings
    eos_seed = 66                               # Seed used for generating the 'eos' symbol's embeddings

    embed_name_bpmn = 'ft_custom'               # Embedding model used for the BPMN sentences
    embed_name_text = 'ft_pretrained'           # Embedding model used for the text sentences


    # Encoder - Decoder
    units = 514                                # Number of units of the LSTM layer
    attention = False                          # Set to 'True' for using the attention mechanism

    # Training
    batch_size = 1                             # Number of sentences in each batch
    epochs = 1                                 # Number of epochs

    # check_files(old_dataset_path)
    input_lang, target_lang, \
    target_tensor, target_lang_tokenizer \
    = load_dataset(paths, num_examples)

    # Creating training and validation sets
    input_lang_train, input_lang_val, \
    target_lang_train, target_lang_val, \
    target_tensor_train, target_tensor_val = \
        train_test_split(input_lang, target_lang,target_tensor,
                         test_size=test_size, random_state=split_seed)


    buffer_size = len(input_lang_train)
    steps_per_epoch = len(input_lang_train) // batch_size
    vocab_target_size = len(target_lang_tokenizer.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_lang_train, target_lang_train, target_tensor_train)).shuffle(buffer_size)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    # -------------------------- EMBEDDING ----------------------------- #

    create_embeddings = dict()

    create_embeddings['ft_custom'] = ft_custom_create
    create_embeddings['ft_pretrained'] = ft_pretrained_create

    embed_file_bpmn = embeddings_dir + embed_name_bpmn + '_bpmn_' \
                  + str(emb_dims) + '_' + str(num_examples) + '.model'

    embed_file_text = embeddings_dir + embed_name_text + '_text_' \
                  + str(emb_dims) + '_' + str(num_examples) + '.model'

    # Create the chosen embedders
    embedder_bpmn = create_embeddings[embed_name_bpmn](target_lang, embed_file_bpmn)
    embedder_text = create_embeddings[embed_name_text](input_lang, embed_file_text)

    print("Embeddings trained")

    # -------------------------- MT ----------------------------- #
    encoder = Encoder(units, batch_size)
    decoder = Decoder(units, batch_size)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # Creates a tensorflow Checkpoint to save the weights
    ckpt_dir = checkpoint_dir + embed_name_bpmn + ('_with_attention' if attention is True else '')
    checkpoint_prefix = os.path.join(ckpt_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    # Restore a previous checkpoint
    if restore is True:
        if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
            if train is True:
                checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
            else:
                checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()

    if train is True:
        # creates the results folder if required
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        filename = results_folder + embed_name_bpmn \
                + ('_with_attention' if attention is True else '') + '_loss.csv'

        # creates the results file. If restoration is enabled and the file already exists, then we do
        # not recreate it, and we will append our results to it.
        if restore is True:
            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    f.write('# mean,stddev\n')

        else:
            with open(filename, 'w') as f:
                    f.write('# mean,stddev\n')



        # callbacks=[tensorboard_callback]
        results = np.zeros((2, 2))

        # training loop
        for epoch in range(epochs):
            epoch_losses = []
            start = time.time()

            enc_hidden = encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (input_lang, target_lang, target_tensor)) in enumerate(dataset.take(steps_per_epoch)):

                # transform input_lang and target_lang into list of list of words
                input_lang = to_list_strings(input_lang.numpy())
                target_lang = to_list_strings(target_lang.numpy())

                # computes the embeddings
                input_embed = embed_all_sentences(input_lang, embedder_text)
                target_embed = embed_all_sentences(target_lang, embedder_bpmn)

                # computes the loss and updates the models
                batch_loss = train_step(input_embed, target_embed, target_tensor, enc_hidden)
                epoch_losses.append(batch_loss)
                total_loss += batch_loss

                if batch % 1 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

            epoch_losses = np.array(epoch_losses)
            results[epoch%2, 0] = epoch_losses.mean()
            results[epoch%2, 1] = epoch_losses.std()

            # saves models and results every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

                with open(filename, 'a') as f:
                    np.savetxt(f, results, delimiter=',')

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


    ' '.join(translate_text_to_graph(input_lang_val[0]))
