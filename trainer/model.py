import tensorflow as tf
import json
import data_helper

def dropout(inputs, keep_prob, is_train=True):
    return tf.cond(tf.cast(is_train, tf.bool), lambda: tf.nn.dropout(inputs, keep_prob),
        lambda: inputs)


class ParsingModel(object):
    """nn dependency parser"""

    def __init__(self, config, batch, word_mat=None, is_train=True):
        self.meta = data_helper.get_data_meta(config)
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=True)
        self.pos_mat = tf.get_variable("pos_mat", shape=[self.meta["pos_class"]+1, config.embed_dim], initializer=tf.truncated_normal_initializer(stddev=0.01), 
            trainable=True)
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.inference(config, batch)

        if is_train:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)     

    def inference(self, config, batch):

        w, p, y = batch.get_next()
        l2_loss = 0

        with tf.variable_scope("embedding"):
            with tf.variable_scope("word"):
                w_emb = tf.nn.embedding_lookup(self.word_mat, w)
                w_emb = tf.reshape(w_emb, shape=[config.batch_size, self.meta["word_feature_dim"] * config.embed_dim])

            with tf.variable_scope("pos"):
                p_emb = tf.nn.embedding_lookup(self.pos_mat, p)
                p_emb = tf.reshape(p_emb, shape=[config.batch_size, self.meta["pos_feature_dim"] * config.embed_dim])
            emb = tf.concat([w_emb, p_emb], axis=1)
            emb_size = emb.get_shape().as_list()[1]
            l2_loss += tf.nn.l2_loss(emb)

        with tf.variable_scope("hidden"):
            W = tf.get_variable("W", shape=[emb_size, config.hidden], initializer=tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable("b", shape=[config.hidden], initializer=tf.constant_initializer(0.))
            h = tf.nn.xw_plus_b(emb, W, b) ** 3
            h = dropout(h, config.keep_prob, self.is_train)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

        with tf.variable_scope("output"):
            W = tf.get_variable("W", shape=[config.hidden, config.out_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.prob = tf.nn.softmax(tf.matmul(h, W))
            l2_loss += tf.nn.l2_loss(W)

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=self.prob)
            losses = loss + config.req_coeff / 2.0 * l2_loss
            self.loss = tf.reduce_mean(losses, name="loss")

            self.y = tf.argmax(self.prob, 1)
            correct_prediction = tf.equal(self.y, tf.argmax(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name="accuracy")


    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

