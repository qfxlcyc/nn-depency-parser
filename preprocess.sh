
# BUCKET_NAME="yq911122-mlengine/toxic_comment"
# DATA_DIR="gs://$BUCKET_NAME/data"

DATA_DIR="./data"

ARC_CLASS_PATH="$DATA_DIR/arc_class.csv"
POS_CLASS_PATH="$DATA_DIR/pos_class.csv"
EMB_PATH="$DATA_DIR/en-cw.txt"
VOCAB_PATH="$DATA_DIR/vocab"

TRAIN_OUTPUT_PATH="$DATA_DIR/train.csv"
TRAIN_INPUT_PATH="$DATA_DIR/train.gold.conll"

DEV_OUTPUT_PATH="$DATA_DIR/dev.csv"
DEV_INPUT_PATH="$DATA_DIR/dev.gold.conll"

TEST_OUTPUT_PATH="$DATA_DIR/test.csv"
TEST_INPUT_PATH="$DATA_DIR/test.gold.conll"

echo "preprocess training data"
python -m preprocess --arc_class_path $ARC_CLASS_PATH \
                     --pos_class_path $POS_CLASS_PATH \
                     --data_path $TRAIN_INPUT_PATH \
                     --emb_path $EMB_PATH \
                     --vocab_path $VOCAB_PATH \
                     --output_path $TRAIN_OUTPUT_PATH \
                     --train

echo "preprocess dev data"
python -m preprocess --arc_class_path $ARC_CLASS_PATH \
                     --pos_class_path $POS_CLASS_PATH \
                     --data_path $DEV_INPUT_PATH \
                     --emb_path $EMB_PATH \
                     --vocab_path $VOCAB_PATH \
                     --output_path $DEV_OUTPUT_PATH \

echo "preprocess test data"
python -m preprocess --arc_class_path $ARC_CLASS_PATH \
                     --pos_class_path $POS_CLASS_PATH \
                     --data_path $TEST_INPUT_PATH \
                     --emb_path $EMB_PATH \
                     --vocab_path $VOCAB_PATH \
                     --output_path $TEST_OUTPUT_PATH \
