

MODEL_NAME=$1

DATA_DIR="./data/"
OUTPUT_DIR="./outputs/"$MODEL_NAME
RBF=0

mkdir -p $OUTPUT_DIR"figs"
mkdir -p $OUTPUT_DIR"figs/rbf"

mkdir -p $OUTPUT_DIR"csv"
mkdir -p $OUTPUT_DIR"csv/rbf"


if [ $RBF -eq 0 ]
then
        FIG_TITLE="AUC_Scores_on_Actual_Data_${MODEL_NAME}"
        PNG_PATH="${OUTPUT_DIR}/figs/${MODEL_NAME}.png"
        CSV_PATH="${OUTPUT_DIR}/csv/${MODEL_NAME}.csv"
        TXT_PATH="${OUTPUT_DIR}/csv/${MODEL_NAME}.txt"
else
        FIG_TITLE="AUC_Scores_on_RBF_Tranformed_Data_${MODEL_NAME}"
        PNG_PATH="${OUTPUT_DIR}/figs/rbf/${MODEL_NAME}.png"
        CSV_PATH="${OUTPUT_DIR}/csv/rbf/${MODEL_NAME}.csv"
        TXT_PATH="${OUTPUT_DIR}/csv/${MODEL_NAME}.txt"
fi


