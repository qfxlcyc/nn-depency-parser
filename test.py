from preprocess import *

def test_load_data():
    path = './data/tmp.gold.conll'
    sentense, pos, arc_labels, trans = load_data(path)
    print sentense
    print pos
    print arc_labels
    print trans
    # [['This', 'time', ',', 'the', 'firms', 'were', 'ready', '.'], ['That', "'s", 'not', 'an', 'average', 'to', 'soothe', 'Giant', 'rooters', '.']]
    # [['DT', 'NN', ',', 'DT', 'NNS', 'VBD', 'JJ', '.'], ['DT', 'VBZ', 'RB', 'DT', 'NN', 'TO', 'VB', 'NNP', 'NNS', '.']]
    # [['det', 'nmod:tmod', 'punct', 'det', 'nsubj', 'cop', 'root', 'punct'], ['nsubj', 'cop', 'neg', 'det', 'root', 'mark', 'acl', 'compound', 'dobj', 'punct']]
    # [[(1, 2), (2, 7), (3, 7), (4, 5), (5, 7), (6, 7), (7, 0), (8, 7)], [(1, 5), (2, 5), (3, 5), (4, 5), (5, 0), (6, 7), (7, 5), (8, 9), (9, 7), (10, 5)]]

def test_write_and_load_class():
    arc_labels = [['det', 'nmod:tmod', 'punct', 'det', 'nsubj', 'cop', 'root', 'punct'], ['nsubj', 'cop', 'neg', 'det', 'root', 'mark', 'acl', 'compound', 'dobj', 'punct']]
    path = './arc_class.csv'
    arc_class = write_class(arc_labels, path)
    loaded_arc_class = load_class(path)
    print arc_class
    print loaded_arc_class
    arc_ids = transform_to_id(arc_labels, arc_class)
    print arc_ids

    pos_labels = [['DT', 'NN', ',', 'DT', 'NNS', 'VBD', 'JJ', '.'], ['DT', 'VBZ', 'RB', 'DT', 'NN', 'TO', 'VB', 'NNP', 'NNS', '.']]
    pos_class = write_class(pos_labels, './pos_class.csv')
    print pos_class
    pos_ids = transform_to_id(pos_labels, pos_class)
    print pos_ids

def test_generating_labels():
    transitions = [
        [(1, 2), (2, 7), (3, 7), (4, 5), (5, 7), (6, 7), (7, 0), (8, 7)], 
        [(1, 5), (2, 5), (3, 5), (4, 5), (5, 0), (6, 7), (7, 5), (8, 9), (9, 7), (10, 5)]
    ]
    for trans in transitions:
        labels = generate_labels_from_arcs(trans)
        print labels
        #[0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 2, 2]
        #[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 2, 0, 0, 1, 2, 0, 2, 2]

def test_word_to_ids():
    sentenses = [['This', 'time', ',', 'the', 'firms', 'were', 'ready', '.'], ['That', "'s", 'not', 'an', 'average', 'to', 'soothe', 'Giant', 'rooters', '.']]
    vocab_processor = get_vocab_processor('./data/en-cw.txt', './data/vocab', create_new=True)
    sentense_ids = transform_sentences(sentenses, vocab_processor)
    print sentense_ids

    vocab_processor = get_vocab_processor('./data/en-cw.txt', './data/vocab', create_new=False)
    sentense_ids = transform_sentences(sentenses, vocab_processor)
    print sentense_ids
    # [[0, 114807, 0, 413, 41010, 123978, 93518, 0], [0, 328, 76501, 644, 450, 434, 106434, 0, 0, 0]]

def test_parser():
    sent = [0, 114807, 0, 413, 41010, 123978, 93518, 0]
    pos = [9, 2, 4, 9, 10, 3, 7, 5]
    arc = [4, 5, 7, 4, 2, 0, 10, 7]
    y = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 2, 2]
    parser = Parser(len(sent))
    # print "parser initial config"
    # print [n.val for n in parser.stack]
    # print [n.val for n in parser.buffer]
    # print [(e[0].val, e[1].val) for e in parser.arc_set]
    # print 

    action_map = {0: "shift", 1: "left-arc", 2: "right-arc"}
    for action in y[2:-1]:   # skip first two 'shift' and last 'left-arc' actions
        parser.step(action)
        print "================="
        print "parser config after taking action %s" % (action_map[action])
        print [n.val for n in parser.stack]
        print [n.val for n in parser.buffer]
        print [(e[0].val, e[1].val) for e in parser.arc_set]
        row = create_features(parser, sent, pos, arc) + [action]
        print "++++++++++++"
        print "features created from the config"
        print row
        print


def main():
    # test_load_data()
    # test_write_and_load_class()
    # test_word_to_ids()
    # test_generating_labels()
    test_parser()


if __name__ == '__main__':
    main()

