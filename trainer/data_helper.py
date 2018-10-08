from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import re
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import collections
import os
import sys

import tensorflow as tf
import numpy as np
from six.moves import cPickle
from tensorflow.contrib import learn

from pandas.compat import StringIO
from tensorflow.python.lib.io import file_io

import json

def load_from_gs(file_path):
    data = []
    with file_io.FileIO(file_path, mode='r') as f:
        lines = f.read().splitlines()
        for d in lines:
            d = d.split(',')
            data.append(d)
    return data

def load_json_from_gs(file_path):
    with file_io.FileIO(file_path, mode='rb') as f:
        data = json.load(f)
    return data

def write_json_to_gs(data, file_path):
    with file_io.FileIO(file_path, mode='wb') as f:
        data = json.dump(data, f)
    return data

def load_word_embedding(file_path):
    embeddings = []
    words = []
    with file_io.FileIO(file_path, mode='r') as f:
        lines = f.read().splitlines()
        for d in lines:
            word, embed_str = d.split('\t')
            embed = [float(e) for e in embed_str.split()]
            embeddings.append(embed)
            # words.append(word)
    # index = vocab_processor.transform(words)
    embeddings.append([0.]*len(embeddings[0]))   # add embedding for Null
    return embeddings


def get_record_parser(config):
    meta = load_json_from_gs(config.meta_file)
    ld, wd, pd, ad = meta["label_dim"], meta["word_feature_dim"], meta["pos_feature_dim"], meta["arc_feature_dim"]
    record_len = wd + pd + ld + ad + 1   # the first col is data id
    # record_len = wd + ld + 1

    def parse(line):
        parsed_line = tf.decode_csv(line, [[0] for _ in xrange(record_len)])

        label = tf.convert_to_tensor(parsed_line[-ld:])
        word_feat = tf.convert_to_tensor(parsed_line[1:wd+1])
        pos_feat = tf.convert_to_tensor(parsed_line[wd+1:wd+pd+1])
        arc_feat = tf.convert_to_tensor(parsed_line[wd+pd+1:-ld])
        return word_feat, pos_feat, arc_feat, label
        # return word_feat, label
    return parse

def get_dataset(text_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TextLineDataset(text_file).map(
        parser, num_parallel_calls=num_threads)
    return dataset

def get_batch_dataset(text_file, parser, config, is_train=True):
    dataset = get_dataset(text_file, parser, config)
    if is_train:
        dataset = dataset.shuffle(config.capacity).batch(config.batch_size, drop_remainder=True).repeat()
    else:
        dataset = dataset.batch(config.batch_size, drop_remainder=True)
    # dataset = dataset.batch(config.batch_size, drop_remainder=True).repeat()
    return dataset


