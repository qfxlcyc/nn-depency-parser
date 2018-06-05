import tensorflow as tf
import numpy as np
import os

from model import ParsingModel
import data_helper
from data_helper import get_record_parser, get_batch_dataset, load_from_gs, load_word_embedding

def train(config):
    word_mat = np.array(load_word_embedding(config.word_emb_file), dtype=np.float32)
    dev_total = 75000

    print("Building model...")
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_file, parser, config)
    dev_dataset = get_batch_dataset(config.dev_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    model = ParsingModel(config, iterator, word_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    lr = config.init_lr

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))

        for _ in xrange(1, config.num_steps + 1):
            global_step = sess.run(model.global_step) + 1

            loss, train_op, accuracy = sess.run([model.loss, model.train_op, model.accuracy], feed_dict={
                                      handle: train_handle})

            if global_step % config.period == 0:
                tf.logging.info("training step: step {} adding summary".format(global_step))
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                accu_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/accuracy", simple_value=accuracy), ])
                writer.add_summary(loss_sum, global_step)

            if global_step % config.checkpoint == 0:
                tf.logging.info("training step: step {} checking the model".format(global_step))
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))

                metrics, summ = evaluate_batch(
                    model, dev_total // config.batch_size + 1, sess, "dev", handle, dev_handle)
                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))

                dev_loss = metrics["loss"]
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(
                    config.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)


def evaluate_batch(model, num_batches, sess, data_type, handle, str_handle):
    losses = []
    accuracies = []
    metrics = {}

    for _ in xrange(1, num_batches + 1):
        loss, accuracy, = sess.run(
            [model.loss, model.accuracy], feed_dict={handle: str_handle})
        losses.append(loss)
        accuracies.append(accuracy)
    loss = np.mean(losses)
    accuracy = np.mean(accuracy)
    metrics["loss"] = loss
    metrics["accuracy"] = accuracy
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    acc_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/acc".format(data_type), simple_value=metrics["accuracy"]), ])
    return metrics, [loss_sum, acc_sum]


# def test(config):
#     word_mat = np.array(load_from_gs(config.word_emb_file), dtype=np.float32)
#     char_mat = np.array(load_from_gs(config.char_emb_file), dtype=np.float32)
#     eval_file = load_from_gs(config.test_eval_file)
#     meta = load_from_gs(config.test_meta)

#     total = meta["total"]

#     print("Loading model...")
#     test_batch = get_dataset(config.test_record_file, get_record_parser(
#         config, is_test=True), config).make_one_shot_iterator()

#     model = ParsingModel(config, test_batch, word_mat, char_mat, trainable=False)

#     sess_config = tf.ConfigProto(allow_soft_placement=True)
#     sess_config.gpu_options.allow_growth = True

#     with tf.Session(config=sess_config) as sess:
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver()
#         saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
#         sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
#         losses = []
#         answer_dict = {}
#         remapped_dict = {}
#         for step in xrange(total // config.batch_size + 1):
#             qa_id, loss, yp1, yp2 = sess.run(
#                 [model.qa_id, model.loss, model.yp1, model.yp2])
#             answer_dict_, remapped_dict_ = convert_tokens(
#                 eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
#             answer_dict.update(answer_dict_)
#             remapped_dict.update(remapped_dict_)
#             losses.append(loss)
#         loss = np.mean(losses)
#         metrics = evaluate(eval_file, answer_dict)
#         write_to_gs(config.answer_file, remapped_dict)
#         print("Exact Match: {}, F1: {}".format(
#             metrics['exact_match'], metrics['f1']))
