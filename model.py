#!/usr/bin/env python3
import operator
import random
import time
import vosk
import numpy as np
import tensorflow as tf

from audio_reader import AudioReader
from file_logger import FileLogger
from utils import FIRST_INDEX, sparse_tuple_from
from utils import convert_inputs_to_ctc_format

from vosk import Model, KaldiRecognizer, SetLogLevel
from vosk import Model as CtcModel
from vosk import KaldiRecognizer as Lstmctc
import sys
import os
import wave
import subprocess
import json

SetLogLevel(0)

# sample rate of auto input features
sample_rate = 8000
num_features = 39 # MFCC features
# Accounting the 0th index +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1 # alphabatical output for final layer

# Hyper-parameters
num_epochs = 100000 # trained using google colab
num_hidden = 256 # hidden layer num_hidden units
batch_size = 16

num_examples = 1
num_batches_per_epoch = 10

audio = AudioReader(audio_dir=None,
                    cache_dir='cache',
                    sample_rate=sample_rate)

# for training log
file_logger = FileLogger('out.tsv', ['curr_epoch', 'train_cost', 'train_ler', 'val_cost', 'val_ler'])

def next_batch(bs=batch_size, train=True):
    # fetches the next batch from the data 
    # used in the training of the asr model
    x_batch = []
    y_batch = []
    original_batch = []
    for k in range(bs):
        ut_length_dict = dict([(k, len(v['target'])) for (k, v) in audio.cache.items()])
        utterances = sorted(ut_length_dict.items(), key=operator.itemgetter(1))
        test_index = 15
        if train:
            utterances = [a[0] for a in utterances[test_index:]]
        else:
            utterances = [a[0] for a in utterances[:test_index]]
        random_utterance = random.choice(utterances)
        training_element = audio.cache[random_utterance]
        target_text = training_element['target']
        if train:
            l_shift = np.random.randint(low=1, high=1000)
            audio_buffer = training_element['audio'][l_shift:]
        else:
            audio_buffer = training_element['audio']
        x, y, seq_len, original = convert_inputs_to_ctc_format(audio_buffer,
                                                               sample_rate,
                                                               target_text,
                                                               num_features)
        x_batch.append(x)
        y_batch.append(y)
        original_batch.append(original)

    y_batch = sparse_tuple_from(y_batch)
    for i, pad in enumerate(np.max(seq_len_batch) - seq_len_batch):
        x_batch[i] = np.pad(x_batch[i], ((0, 0), (0, pad), (0, 0)), mode='constant', constant_values=0)

    x_batch = np.concatenate(x_batch, axis=0)

    return x_batch, y_batch, original_batch

def decode_batch(d, original, phase='training'):
    # we encode char to ascii before training, decoding here
    aligned_original_string = ''
    aligned_decoded_string = ''
    for jj in range(batch_size)[0:2]:  # just for visualisation purposes. we display only 2.
        values = d.values[np.where(d.indices[:, 0] == jj)[0]]
        str_decoded = ''.join([chr(x) for x in np.asarray(values) + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
        maxlen = max(len(original[jj]), len(str_decoded))
        aligned_original_string += str(original[jj]).ljust(maxlen) + ' | '
        aligned_decoded_string += str(str_decoded).ljust(maxlen) + ' | '
    print('- Original (%s) : %s ...' % (phase, aligned_original_string))
    print('- Decoded  (%s) : %s ...' % (phase, aligned_decoded_string))

def build_or_load_model(reload_model=False, *args, **kwargs):
    if args:
        build_ctc()
    return CtcModel


def build_ctc():
    graph = tf.Graph()
    with graph.as_default():
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_step_size, num_features], but the
        # batch_size and max_step_size can vary along each step
        inputs = tf.compat.v1.placeholder(tf.float32, [None, None, num_features], name='inputs')
        targets = tf.compat.v1.sparse_placeholder(tf.int32, name='targets')

        # 1d array of size [batch_size]
        seq_len = tf.compat.v1.placeholder(tf.int32, [None], name='seq_len')

        # Defining the cell
        # Can be:
        #   tf.nn.rnn_cell.RNNCell
        #   tf.nn.rnn_cell.GRUCell
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

        # Stacking rnn cells
        stack = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell], state_is_tuple=True)

        # The second output is the last state and we will no use that
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

        shape = tf.shape(input=inputs)
        batch_s, max_time_steps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # Truncated normal with mean 0 and stdev=0.1
        W = tf.Variable(tf.random.truncated_normal([num_hidden,
                                             num_classes],
                                            stddev=0.1))
        # Zero initialization
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        logits = tf.transpose(a=logits, perm=(1, 0, 2))

        loss = tf.compat.v1.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(input_tensor=loss)

        # optimizer = tf.train.AdamOptimizer().minimize(cost)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=5e-4).minimize(cost)

        # using greedy decoder
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

        # Inaccuracy: label error rate
        ler = tf.reduce_mean(input_tensor=tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))

    with tf.compat.v1.Session(graph=graph) as session:

        tf.compat.v1.global_variables_initializer().run()

        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            start = time.time()

            for batch in range(num_batches_per_epoch):
                train_inputs, train_targets, train_seq_len, original = next_batch(train=True)
                feed = {inputs: train_inputs,
                        targets: train_targets,
                        seq_len: train_seq_len}

                batch_cost, _, train_ler_p, d = session.run([cost, optimizer, ler, decoded[0]], feed)
                train_cost += batch_cost / num_batches_per_epoch
                train_ler += train_ler_p / num_batches_per_epoch
                decode_batch(d, original, phase='training')

            val_inputs, val_targets, val_seq_len, val_original = next_batch(train=False)
            val_feed = {inputs: val_inputs,
                        targets: val_targets,
                        seq_len: val_seq_len}

            val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

            # Decoding
            # np.where(np.diff(d.indices[:, 0]) == 1)
            d = session.run(decoded[0], feed_dict=val_feed)
            decode_batch(d, val_original, phase='validation')

            print('-' * 80)
            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, " \
                  "val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"

            file_logger.write([curr_epoch + 1,
                               train_cost,
                               train_ler,
                               val_cost,
                               val_ler])

            print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
                             val_cost, val_ler, time.time() - start))


if _name_ == '_main_':
    build_ctc()

    if not os.path.exists("model"):
        exit (1)

    sample_rate=16000
    model = CtcModel("model")
    rec = Lstmctc(model, sample_rate)

    process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                                sys.argv[1],
                                '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
                                stdout=subprocess.PIPE)

    previousData = ""
    finalData = ""
    while True:
        data = process.stdout.read(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())["text"]
            print(result)
        else:
            partialResult = json.loads(rec.PartialResult())["partial"]
            print(partialResult, end="\r")
        # finalData += result

# print(finalData)