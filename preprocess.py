import tensorflow as tf
import numpy as np
import argparse
import trainer.data_helper as data_helper
from parser import Parser
import os

OOV = "--OOV--"
NULL = "--NULL--"

def load_data(path):
    """ example of a record:
    1   The _   DET DT  _   4   det _   _
    2   complex _   ADJ JJ  _   4   amod    _   _
    3   financing   _   NOUN    NN  _   4   compound    _   _
    4   plan    _   NOUN    NN  _   10  nsubj   _   _
    5   in  _   ADP IN  _   9   case    _   _
    6   the _   DET DT  _   9   det _   _
    7   S&L _   NOUN    NN  _   9   compound    _   _
    8   bailout _   NOUN    NN  _   9   compound    _   _
    9   law _   NOUN    NN  _   4   nmod    _   _
    10  includes    _   VERB    VBZ _   0   root    _   _
    11  raising _   VERB    VBG _   10  xcomp   _   _
    12  $   _   SYM $   _   11  dobj    _   _
    13  30  _   NUM CD  _   14  compound    _   _
    14  billion _   NUM CD  _   12  nummod  _   _
    15  from    _   ADP IN  _   16  case    _   _
    16  debt    _   NOUN    NN  _   11  nmod    _   _
    17  issued  _   VERB    VBN _   16  acl _   _
    18  by  _   ADP IN  _   22  case    _   _
    19  the _   DET DT  _   22  det _   _
    20  newly   _   ADV RB  _   21  advmod  _   _
    21  created _   VERB    VBN _   22  amod    _   _
    22  RTC _   PROPN   NNP _   17  nmod    _   _
    23  .   _   PUNCT   .   _   10  punct   _   _
    """
    with open(path, 'r') as f:
        data = f.read().split('\n\n')
        data = [record.split('\n') for record in data]
        data = [[l.split('\t') for l in record] for record in data]
        
        sentence, pos, arc_labels, trans = [], [], [], []
        for record in data:
            min_len = min(len(l) for l in record)
            if min_len < 7:
                print "record with invalid line"
                print record
                continue
            sentence.append([l[1].lower() for l in record])
            pos.append([l[4] for l in record])
            arc_labels.append([l[7] for l in record])
            trans.append([(int(l[0]), int(l[6])) for l in record])
        return sentence, pos, arc_labels, trans


def load_embed(path):
    with open(path, 'r') as f:
        embed = f.readlines()
        return embed

def write_embed(path, embed):
    with open(path, 'w') as f:
        for e in embed:
            f.write(e)


class Preprocess:

    def __init__(self, config):
        self.config = config

    def load(self, file_path):
        self.sentences, self.pos, self.arc_labels, self.arcs = load_data(file_path)

    def fit(self):
        config = self.config

        arc2id_dict = create_item2id_dict(self.arc_labels)
        pos2id_dict = create_item2id_dict(self.pos)
        data_helper.write_json_to_gs(arc2id_dict, config.arc2id_file)
        data_helper.write_json_to_gs(pos2id_dict, config.pos2id_file)

        embed = load_embed(config.embedding_file)

        print "convert words to indices"
        word2id_dict = create_word2id_dict(embed)
    
        # print "add --OOV-- and --NULL-- to embeddings"
        print "add --NULL-- to embeddings"
        embed_size = len(embed[0].split(' '))
        print embed_size
        # added = ['\t'.join(["%s" % (word)] + [' '.join([str(0.)]*embed_size)]) + '\n' for word in [OOV, NULL]]
        added = ['\t'.join(["%s" % (NULL)] + [' '.join([str(0.)]*embed_size)]) + '\n']
        write_embed(config.embedding_file, added + embed)

        print "create id2word dict for prediction use"
        id2word_dict = {i:w for w, i in word2id_dict.items()}
        data_helper.write_json_to_gs(id2word_dict, config.id2word_file)

        data_helper.write_json_to_gs(word2id_dict, config.word2id_file)

        print "create label2id dict"
        label2id_dict = create_ulabel2id_dict(arc2id_dict)
        data_helper.write_json_to_gs(label2id_dict, config.label2id_file)

        print "create meta file"
        meta = {
            "label_dim": len(label2id_dict),
            "word_feature_dim": 18,
            "pos_feature_dim": 18,
            "arc_feature_dim": 12,
            "num_pos_class": len(pos2id_dict),
            "num_arc_class": len(arc2id_dict)
        }
        data_helper.write_json_to_gs(meta, config.meta_file)

    def transform(self, output_path):
        config = self.config

        arc2id_dict = data_helper.load_json_from_gs(config.arc2id_file)
        pos2id_dict = data_helper.load_json_from_gs(config.pos2id_file)
        label2id_dict = data_helper.load_json_from_gs(config.label2id_file)
        word2id_dict = data_helper.load_json_from_gs(config.word2id_file)

        print "convert arc labels, pos and words to indexes"
        arc_labels = list2id(self.arc_labels, arc2id_dict)
        pos = list2id(self.pos, pos2id_dict)
        sentences = list2id(self.sentences, word2id_dict)

        print "generate labels"
        parser = Parser(skip2shifts=False)
        labels = [parser.create_labels(arc, arc_la, label2id_dict) for arc, arc_la in zip(self.arcs, self.arc_labels)]

        print "create dataset"
        create_data_for_model(output_path, parser, sentences, pos, arc_labels, labels, label2id_dict)

def _create_x2id_dict_with_default(item_set):
    x2id_dict = {e:i for i, e in enumerate(item_set, 1)}
    # x2id_dict[OOV] = 0
    x2id_dict[NULL] = 0
    return x2id_dict

def create_word2id_dict(embed):
    
    def extract_word(e):
        return e.split('\t')[0]

    vocab = [extract_word(e) for e in embed]
    return _create_x2id_dict_with_default(set(vocab))

def create_label2id_dict(arc2id_dict):
    label2id_dict = {'SHIFT': 0}
    i = 0
    for k in arc2id_dict:
        # if k in [OOV, NULL]: continue
        if k == NULL: continue
        for j, direction in enumerate(['LEFT', 'RIGHT']):
            label2id_dict["%s_%s" % (direction, k)] = 2 * i + j + 1
        i += 1
    return label2id_dict

def create_ulabel2id_dict(arc2id_dict):
    label2id_dict = {'SHIFT': 0, 'LEFT': 1, 'RIGHT': 2}
    return label2id_dict

def create_item2id_dict(items):
    item_set = set()
    for l in items:
        item_set = item_set.union(set(l))
    return _create_x2id_dict_with_default(item_set)

def list2id(lists, item2id_dict, default=0):
    list_ids = []
    for l in lists:
        list_ids.append([item2id_dict.get(i, default) for i in l])
    return list_ids


def create_data_for_model(data_path, parser, sentence_ids, pos_ids, arc_ids, labels, label2id_dict):
    with open(data_path, 'w') as f:
        count = 0
        num_class = len(label2id_dict)
        for i, (sent, pos, arc, la) in enumerate(zip(sentence_ids, pos_ids, arc_ids, labels)):
            parser.fit(len(sent))
            # for l in la[2:-1]:   # skip first two 'shift' and last 'right-arc' actions
            for l in la:
                parser.step(l)

                one_hot_label = [0]*num_class
                one_hot_label[l] = 1
                row = [i] + parser.create_features(sent, pos, arc) + one_hot_label
                # print row
                f.write(','.join(str(e) for e in row)+'\n')
            count += 1
            if count % 100 == 0:
                print "created data from %s sentences" % (count)


if __name__ == '__main__':
    project_dir = "./"
    data_dir = os.path.join(project_dir, "data")

    arc2id_file = os.path.join(data_dir, "arc2id.json")
    pos2id_file = os.path.join(data_dir, "pos2id.json")
    word2id_file = os.path.join(data_dir, "word2id.json")
    id2word_file = os.path.join(data_dir, "id2word.json")
    label2id_file = os.path.join(data_dir, "label2id.json")
    embedding_file = os.path.join(data_dir, "en-cw.txt")
    meta_file = os.path.join(data_dir, "meta.json")

    train_file = os.path.join(data_dir, "train.gold.conll")
    dev_file = os.path.join(data_dir, "dev.gold.conll")
    test_file = os.path.join(data_dir, "test.gold.conll")
    
    train_output_file = os.path.join(data_dir, "train")
    dev_output_file = os.path.join(data_dir, "dev")
    test_output_file = os.path.join(data_dir, "test")

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--arc2id_file', type=str, default=arc2id_file,
                        help='file storing arc class')
    parser.add_argument('--pos2id_file', type=str, default=pos2id_file,
                        help='file storing pos class')
    parser.add_argument('--word2id_file', type=str, default=word2id_file,
                        help='file storing arc class')
    parser.add_argument('--id2word_file', type=str, default=id2word_file,
                        help='file storing pos class')
    parser.add_argument('--label2id_file', type=str, default=label2id_file,
                        help='file storing pos class')
    parser.add_argument('--embedding_file', type=str, default=embedding_file,
                        help='file storing embeddings')
    parser.add_argument('--meta_file', type=str, default=meta_file,
                        help='file storing embeddings')
    
    parser.add_argument('--data_file', type=str, default=train_file,
                        help='input data for preprocessing')
    parser.add_argument('--output_file', type=str, default=train_output_file,
                        help='input data for preprocessing')
    parser.add_argument('--fit', action="store_true",
                        help='input data for preprocessing')
    args = parser.parse_args()

    proc = Preprocess(args)

    print "load data from %s" % (args.data_file)
    proc.load(args.data_file)

    if args.fit:
        print "write x2id files"
        proc.fit()

    print "write transformed data to %s" % (args.output_file)
    proc.transform(args.output_file)







