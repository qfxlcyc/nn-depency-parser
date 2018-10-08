import tensorflow as tf
import json
import data_helper

def dropout(inputs, keep_prob, is_train=True):
    if is_train: return tf.nn.dropout(inputs, keep_prob)
    else: return inputs

class ParsingModel(object):
    """nn dependency parser"""

    def __init__(self, config, batch, word_mat=None, is_train=True):
        self.meta = data_helper.load_json_from_gs(config.meta_file)
        self.is_train = is_train
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=True)
        self.pos_mat = tf.get_variable("pos_mat", shape=[self.meta["num_pos_class"], config.embed_dim], initializer=tf.random_uniform_initializer(-0.01, 0.01), trainable=True)
        self.arc_mat = tf.get_variable("arc_mat", shape=[self.meta["num_arc_class"], config.embed_dim], initializer=tf.random_uniform_initializer(-0.01, 0.01), trainable=True)
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.build(config, batch)

        if is_train:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            # self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-6)
            self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)
            # beta1_power, beta2_power = self.opt._get_beta_accumulators()
            # self._lr = (self.opt._lr_t * tf.sqrt(1 - beta1_power) / (1 - beta2_power))

            summ = []
            for g, v in grads:
                if g is not None:
                    #print(format(v.name))
                    grad_hist_summary = tf.summary.histogram("{}/grad_histogram".format(v.name), g)
                    # sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    mean_summary = tf.summary.scalar("{}/grad_mean".format(v.name), tf.reduce_mean(g))
                    summ.append(grad_hist_summary)
                    # summ.append(sparsity_summary)
                    # summ.append(mean_summary)
                    summ.append(mean_summary)
            self.grad_summ = tf.summary.merge(summ)

        self.saver = tf.train.Saver()  

    def build(self, config, batch):

        w, p, a, y = batch.get_next()
        # w, y = batch.get_next()
        l2_loss = 0.

        with tf.variable_scope("embedding"):
            with tf.variable_scope("word"):
                w_emb = tf.nn.embedding_lookup(self.word_mat, w)
                w_emb_size = self.meta["word_feature_dim"] * config.embed_dim
                w_emb = tf.reshape(w_emb, shape=[config.batch_size, w_emb_size])
                l2_loss += tf.nn.l2_loss(w_emb)

            with tf.variable_scope("pos"):
                p_emb = tf.nn.embedding_lookup(self.pos_mat, p)
                p_emb_size = self.meta["pos_feature_dim"] * config.embed_dim
                p_emb = tf.reshape(p_emb, shape=[config.batch_size, p_emb_size])
                l2_loss += tf.nn.l2_loss(p_emb)

            with tf.variable_scope("arc"):
                a_emb = tf.nn.embedding_lookup(self.arc_mat, a)
                a_emb_size = self.meta["arc_feature_dim"] * config.embed_dim
                a_emb = tf.reshape(a_emb, shape=[config.batch_size, a_emb_size])
                l2_loss += tf.nn.l2_loss(a_emb)
            embeddings = [w_emb, p_emb, a_emb]

        with tf.variable_scope("hidden"):
            W_w = tf.get_variable("W_word", shape=[w_emb_size, config.hidden], initializer=tf.truncated_normal_initializer(stddev=0.01))
            W_p = tf.get_variable("W_pos", shape=[p_emb_size, config.hidden], initializer=tf.truncated_normal_initializer(stddev=0.01))
            W_a = tf.get_variable("W_arc", shape=[a_emb_size, config.hidden], initializer=tf.truncated_normal_initializer(stddev=0.01))
            W = [W_w, W_p, W_a]
            b = tf.get_variable("b1", shape=[config.hidden], initializer=tf.constant_initializer(0.))
            h = tf.pow(tf.add_n([tf.matmul(emb, w) for emb, w in zip(embeddings, W)]) + b, 3)
            h = dropout(h, config.keep_prob, self.is_train)
            l2_loss += tf.nn.l2_loss(W_w)

        with tf.variable_scope("output"):
            W2 = tf.get_variable("W2", shape=[config.hidden, self.meta["label_dim"]], initializer=tf.truncated_normal_initializer(stddev=0.01))
            b2 = tf.get_variable("b2", shape=[self.meta["label_dim"]], initializer=tf.constant_initializer(0.))
            self.prob = tf.nn.softmax(tf.nn.xw_plus_b(h, W2, b2))
            l2_loss += tf.nn.l2_loss(W2)

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=self.prob)
            self.loss = tf.reduce_mean(loss) + config.req_coeff / 2.0 * tf.reduce_sum(l2_loss)

            self.y = tf.argmax(self.prob, 1)
            correct_prediction = tf.equal(self.y, tf.argmax(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name="accuracy")


    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

