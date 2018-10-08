from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.lib.io import file_io
from trainer.parser import Parser

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
    embeddings.append([0.]*len(embeddings[0]))   # add embedding for NULL
    return embeddings


def get_record_parser(config):
    meta = load_json_from_gs(config.meta_file)
    ld, wd, pd, ad = meta["label_dim"], meta["word_feature_dim"], meta["pos_feature_dim"], meta["arc_feature_dim"]
    record_len = wd + pd + ld + ad + 1   # the first col is data id
    # record_len = wd + ld + 1

    def parse(line):
        parsed_line = tf.decode_csv(line, [[0] for _ in xrange(record_len)])

        qid = tf.convert_to_tensor(parsed_line[0])
        label = tf.convert_to_tensor(parsed_line[-ld:])
        word_feat = tf.convert_to_tensor(parsed_line[1:wd+1])
        pos_feat = tf.convert_to_tensor(parsed_line[wd+1:wd+pd+1])
        arc_feat = tf.convert_to_tensor(parsed_line[wd+pd+1:-ld])
        return qid, word_feat, pos_feat, arc_feat, label
        # return word_feat, label
    return parse

def get_dataset(text_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TextLineDataset(text_file).map(
        parser, num_parallel_calls=num_threads)
    return dataset

def get_batch_dataset(text_file, parser, config, batch_size, is_train=True):
    dataset = get_dataset(text_file, parser, config)
    if is_train:
        dataset = dataset.shuffle(config.capacity).batch(batch_size, drop_remainder=True).repeat()
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    # dataset = dataset.batch(config.batch_size, drop_remainder=True).repeat()
    return dataset


def get_sentence_length(actions):
    return sum(1 for a in actions if a != Parser.SHIFT)

def uas_count(actual_arc, pred_arc):
    actual_arc_dict = {end:start for start, end in actual_arc}
    pred_arc_dict = {end:start for start, end in pred_arc}
    return sum(1 for k in actual_arc_dict if actual_arc_dict[k] == pred_arc_dict.get(k, None))

def unlabeled_attachment_score(actual, prediction):
    scores = {}
    parser = Parser()
    for qid in actual:
        actual_act, pred_act = actual[qid], prediction[qid]
        sent_len = get_sentence_length(actual_act)
        if sent_len == 0: continue
        parser.fit(sent_len)
        actual_arc = parser.actions_to_arcs(actual_act)
        pred_arc = parser.actions_to_arcs(pred_act)
        scores[qid] = uas_count(actual_arc, pred_arc) / float(sent_len)
    return scores




