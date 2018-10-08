
import tensorflow as tf
import numpy as np
import os

from model import ParsingModel
from train_helper import *
import data_helper
from collections import defaultdict

def train(config):
    word_mat = np.array(data_helper.load_word_embedding(config.word_emb_file), dtype=np.float32)

    print("Building model...")
    # data_manager = DataManager(config)

    train_graph = tf.Graph()
    dev_graph = tf.Graph()

    parser = data_helper.get_record_parser(config)
    train_dataset = data_helper.get_batch_dataset(config.train_file, parser, config, config.batch_size)
    dev_dataset = data_helper.get_batch_dataset(config.dev_file, parser, config, config.eval_batch_size, is_train=False)

    # initialize train model and dev model separately
    with train_graph.as_default():
        train_iterator_manager = IteratorManager(train_dataset)
        train_model = ParsingModel(config, train_iterator_manager.iterator, word_mat)
        initializer = tf.global_variables_initializer()

    with dev_graph.as_default():
        dev_iterator_manager = IteratorManager(dev_dataset)
        dev_model = ParsingModel(config, dev_iterator_manager.iterator, word_mat, is_train=False)

    checkpoints_path = os.path.join(config.save_dir, "checkpoints")

    # initialize train and dev session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    train_sess = tf.Session(graph=train_graph, config=sess_config)
    dev_sess = tf.Session(graph=dev_graph, config=sess_config)

    train_sess.run(initializer)
    train_iterator_manager.get_string_handle(train_sess)
    dev_iterator_manager.get_string_handle(dev_sess)

    summary_writer = SummaryWriter(config.log_dir)

    lr_updater = LearningRateUpdater(patience=3, init_lr=config.init_lr, loss_save=100.0)
    lr_updater.assign(train_sess, train_model)

    # checkpoint_path = tf.train.latest_checkpoint(config.save_dir, latest_filename=None)
    # train_model.saver.restore(train_sess, checkpoint_path)
    
    for _ in xrange(1, config.num_steps + 1):

        global_step = train_sess.run(train_model.global_step) + 1

        loss, accuracy, train_op, grad_summ = train_sess.run([train_model.loss, train_model.accuracy, train_model.train_op, train_model.grad_summ], feed_dict=train_iterator_manager.make_feed_dict())

        if global_step % config.period == 0:
            tf.logging.info("training step: step {} adding loss: {}".format(global_step, loss))
            summ = model_summary('model', loss, accuracy)
            summ += [grad_summ]
            summary_writer.write_summaries(summ, global_step)
            summary_writer.flush()
                        

        if global_step % config.checkpoint == 0:
            # lr_updater.setZero(train_sess, train_model)
            tf.logging.info("training step: step {} checking the model".format(global_step))
            checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=global_step)

            # summ = evaluate_batch(train_model, config.val_num_batches, train_sess, "train", train_iterator_manager)
            # summary_writer.write_summaries(summ, global_step)
            
            dev_model.saver.restore(dev_sess, checkpoint_path)
            summ = evaluate_batch(dev_model, config.dev_val_num_sentences, dev_sess, "dev", dev_iterator_manager)
            summary_writer.write_summaries(summ, global_step)
            
            summary_writer.flush()
    test(config)

def test(config):
    word_mat = np.array(data_helper.load_word_embedding(config.word_emb_file), dtype=np.float32)
    test_graph = tf.Graph()

    parser = data_helper.get_record_parser(config)
    test_dataset = data_helper.get_batch_dataset(config.test_file, parser, config, config.eval_batch_size, is_train=False)

    with test_graph.as_default():
        test_iterator_manager = IteratorManager(test_dataset)
        test_model = ParsingModel(config, test_iterator_manager.iterator, word_mat, is_train=False)

    checkpoints_path = os.path.join(config.save_dir, "checkpoints")

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    test_sess = tf.Session(graph=test_graph, config=sess_config)
    test_iterator_manager.get_string_handle(test_sess)

    summary_writer = SummaryWriter(config.log_dir)

    checkpoint_path = tf.train.latest_checkpoint(config.save_dir, latest_filename=None)
    test_model.saver.restore(test_sess, checkpoint_path)
    summ = evaluate_batch(test_model, config.test_val_num_sentences, test_sess, "test", test_iterator_manager)
    summary_writer.write_summaries(summ, 0)
    summary_writer.flush()


def model_summary(data_type, loss, accuracy, uas=None):
    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="%s/loss" % (data_type), simple_value=loss), ])
    accuracy_sum = tf.Summary(value=[tf.Summary.Value(tag="%s/accuracy" % (data_type), simple_value=accuracy), ])
    summ = [loss_sum, accuracy_sum]
    if uas is not None:
        uas_sum = tf.Summary(value=[tf.Summary.Value(tag="%s/uas" % (data_type), simple_value=uas), ])
        summ.append(uas_sum)
    return summ


def evaluate_batch(model, num_sentences, sess, data_type, data_manager):
    losses = []
    accuracies = []
    actual = defaultdict(list)
    pred = defaultdict(list)
    qids = set()
    while len(qids) <= num_sentences:
        try:
            qid, loss, accuracy, y, y_hat = sess.run([model.qid, model.loss, model.accuracy, model.y, model.y_hat], feed_dict=data_manager.make_feed_dict())
            qid_set = np.unique(qid)

            for i in qid_set:
                actual[i] += y[qid==i].tolist()
                pred[i] += y_hat[qid==i].tolist()
            qids = qids.union(set(qid_set.tolist()))
        except:
            tf.logging.info("Error in evaluation")
            break
        losses.append(loss)
        accuracies.append(accuracy)

    loss = np.mean(losses)
    accuracy = np.mean(accuracies)
    uas_by_qid = data_helper.unlabeled_attachment_score(actual, pred)
    uas = np.mean(uas_by_qid.values())
    tf.logging.info("Evaluation: loss: {} accuracy: {} uas: {}".format(loss, accuracy, uas))
    # print "Evaluation: loss: {} accuracy: {} uas: {}".format(loss, accuracy, uas)
    return model_summary(data_type, loss, accuracy, uas)

