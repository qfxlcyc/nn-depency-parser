
# BUCKET_NAME="yq911122-mlengine/toxic_comment"
# DATA_DIR="gs://$BUCKET_NAME/data"

DATA_DIR="./data"

DEBUG_OUTPUT_FILE="$DATA_DIR/debug"
DEBUG_INPUT_FILE="$DATA_DIR/tmp.gold.conll"

TRAIN_OUTPUT_FILE="$DATA_DIR/train"
TRAIN_INPUT_FILE="$DATA_DIR/train.gold.conll"

DEV_OUTPUT_FILE="$DATA_DIR/dev"
DEV_INPUT_FILE="$DATA_DIR/dev.gold.conll"

TEST_OUTPUT_FILE="$DATA_DIR/test"
TEST_INPUT_FILE="$DATA_DIR/test.gold.conll"

# echo "preprocess debugging data"
# python -m preprocess --data_file $DEBUG_INPUT_FILE \
#                      --output_file $DEBUG_OUTPUT_FILE \
#                      --fit

echo "preprocess training data"
python -m preprocess --data_file $TRAIN_INPUT_FILE \
                     --output_file $TRAIN_OUTPUT_FILE \
                     --fit

echo "preprocess dev data"
python -m preprocess --data_file $DEV_INPUT_FILE \
                     --output_file $DEV_OUTPUT_FILE \

echo "preprocess test data"
python -m preprocess --data_file $TEST_INPUT_FILE \
                     --output_file $TEST_OUTPUT_FILE \
