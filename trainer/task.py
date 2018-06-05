import os
import tensorflow as tf

from main import train

flags = tf.flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project_dir = "./"
data_dir = os.path.join(project_dir, "data")
log_dir = os.path.join(project_dir, "log/event")
save_dir = os.path.join(project_dir, "log/model")

train_file = os.path.join(data_dir, "train.csv")
dev_file = os.path.join(data_dir, "dev.csv")
test_file = os.path.join(data_dir, "test.csv")
word_emb_file = os.path.join(data_dir, "en-cw.txt")
meta_file = os.path.join(data_dir, "meta.json")

flags.DEFINE_string("mode", "train", "train/debug/test")

flags.DEFINE_string("data_dir", data_dir, "")
flags.DEFINE_string("log_dir", log_dir, "")
flags.DEFINE_string("save_dir", save_dir, "")

flags.DEFINE_string("train_file", train_file, "")
flags.DEFINE_string("dev_file", dev_file, "")
flags.DEFINE_string("test_file", test_file, "")
flags.DEFINE_string("word_emb_file", word_emb_file, "")
flags.DEFINE_string("meta_file", meta_file, "")

flags.DEFINE_integer("embed_dim", 50, "Embedding dimension")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")

flags.DEFINE_integer("hidden", 200, "Hidden layer dimension")
flags.DEFINE_integer("out_dim", 3, "Parser output dimension")

flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_steps", 10000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint for evaluation")
flags.DEFINE_integer("period", 50, "period to save batch loss")
flags.DEFINE_float("init_lr", 0.01, "Initial lr for Adadelta")
flags.DEFINE_float("req_coeff", 10e-8, "coefficient of l2 loss")
flags.DEFINE_float("keep_prob", 0.5, "Keep prob in model")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("patience", 3, "Patience for lr decay")


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "debug":
        config.train_file = os.path.join(data_dir, "debug.csv")
        config.dev_file = os.path.join(data_dir, "debug.csv")
        config.num_steps = 2
        config.checkpoint = 2
        config.period = 1
        train(config)
    # elif config.mode == "test":
    #     test(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
