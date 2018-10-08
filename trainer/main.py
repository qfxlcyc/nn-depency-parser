
import tensorflow as tf
import numpy as np
import os

from model import ParsingModel
from train_helper import *
import data_helper


def train(config):
    word_mat = np.array(data_helper.load_word_embedding(config.word_emb_file), dtype=np.float32)

    print("Building model...")
    # data_manager = DataManager(config)

    train_graph = tf.Graph()
    dev_graph = tf.Graph()

    parser = data_helper.get_record_parser(config)
    train_dataset = data_helper.get_batch_dataset(config.train_file, parser, config)
    dev_dataset = data_helper.get_batch_dataset(config.dev_file, parser, config, is_train=False)

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
            summ = evaluate_batch(dev_model, config.val_num_batches, dev_sess, "dev", dev_iterator_manager)
            summary_writer.write_summaries(summ, global_step)
            
            summary_writer.flush()
                        
    # tf.logging.info("training step: step {} checking the model".format(global_step))
    # checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=global_step)

    # dev_model.saver.restore(dev_sess, checkpoint_path)
    # summ = evaluate_batch(dev_model, config.val_num_batches, dev_sess, "dev", dev_iterator_manager)
    # summary_writer.write_summaries(summ, global_step)
    # summary_writer.flush()

def model_summary(data_type, loss, accuracy, lr=None):
    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="%s/loss" % (data_type), simple_value=loss), ])
    accuracy_sum = tf.Summary(value=[tf.Summary.Value(tag="%s/accuracy" % (data_type), simple_value=accuracy), ])
    summ = [loss_sum, accuracy_sum]
    if lr is not None:
        lr_sum = tf.Summary(value=[tf.Summary.Value(tag="%s/learning_rate" % (data_type), simple_value=lr), ])
        summ.append(lr_sum)
    return summ

def evaluate_batch(model, num_batches, sess, data_type, data_manager):
    losses = []
    accuracies = []
    for step in xrange(1, num_batches + 1):
        try:
            loss, accuracy = sess.run([model.loss, model.accuracy], feed_dict=data_manager.make_feed_dict())
        except:
            tf.logging.info("Error in evaluating step: step {}".format(step))
            break
        losses.append(loss)
        accuracies.append(accuracy)
    loss = np.mean(losses)
    accuracy = np.mean(accuracies)
    tf.logging.info("Evaluation: loss: {} accuracy: {}".format(loss, accuracy))
    return model_summary(data_type, loss, accuracy)

