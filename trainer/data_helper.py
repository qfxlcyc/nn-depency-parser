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


def get_data_meta(config):
    # with open(config.meta_file, 'r') as f:
    #     meta = json.load(f)
    with file_io.FileIO(config.meta_file, mode='r') as f:
        meta = json.load(f)
    return meta

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

def get_vocab(vocab_path):
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    return vocab_processor

def load_class(path):
    with open(path, 'r') as f:
        class_map = {}
        for r in f.read().splitlines():
            v, k = r.split(',', 1)
            class_map[k] = int(v)
    return class_map

class Node:

    def __init__(self, val=-1):
        self.val = val
        self.left_chidren = []
        self.right_children = []

    def add_child(self, child, left=False):
        # the later a child is added, the more left/right-most the child is.
        if left:
            self.left_chidren.append(child)
        else:
            self.right_children.append(child)

    def nth_child(self, n, left=False):
        if left:
            return self.left_chidren[-n] if n <= len(self.left_chidren) else Node()
        else:
            return self.right_children[-n] if n <= len(self.right_children) else Node()


class Parser:

    def __init__(self, buffer_size):
        # auto add first two items (if any) to stack to skip the first two shift actions
        self.stack = [Node()] + [Node(i) for i in xrange(min(2, buffer_size))]
        self.buffer = [Node(i) for i in xrange(buffer_size-1, 1, -1)]
        self.arc_set = []

    def step(self, action):
        """execuate action and update stack, buffer and arc-set"""
        s, a, b = self.stack, self.arc_set, self.buffer
        if action == 0: # shift
            s.append(b.pop())
        elif action == 1:   # left-arc
            t = s.pop(len(s)-2)
            c = s[-1]
            a.append((t, c))
            c.add_child(t, left=True)

        else: #right-arc
            t = s.pop()
            c = s[-1]
            a.append((t, c))
            c.add_child(t, left=False)

def get_record_parser(config):
    meta = get_data_meta(config)
    ld, wd, pd = meta["label_dim"], meta["word_feature_dim"], meta["pos_feature_dim"]

    def convert_for_embedding_lookup(l, embed_size):
        return [tf.cond(tf.greater(e, tf.constant(0)),
            lambda: e, lambda: tf.constant(embed_size)) for e in l]

    def parse(line):
        parsed_line = tf.decode_csv(line, [[0] for _ in xrange(meta["input_dim"])])
        label = parsed_line[-ld:]
        word_features = convert_for_embedding_lookup(parsed_line[ : wd ], meta["embed_size"])
        pos_features = convert_for_embedding_lookup(parsed_line[ wd : wd + pd ], meta["pos_class"])

        label = tf.convert_to_tensor(label)
        word_features = tf.convert_to_tensor(word_features)
        pos_features = tf.convert_to_tensor(pos_features)
        return word_features, pos_features, label
    return parse


def get_batch_dataset(text_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TextLineDataset(text_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()

    dataset = dataset.batch(config.batch_size)
    return dataset
