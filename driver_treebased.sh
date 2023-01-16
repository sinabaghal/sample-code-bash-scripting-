#!/bin/sh
echo "Implemented tree based models: xgboost, catboost, rf"

MODEL_NAME=$1 
source ./setup.sh $MODEL_NAME


python src/treebased.py \
--start_depth 14 --end_depth 15 --step_depth 1 \
--start_n_es 395 --end_n_es 400 --step_n_es 5 \
--learning_rate 1e-2 \
--model_name $MODEL_NAME \
--output_dir $OUTPUT_DIR \
--rbf $RBF \
--data_dir $DATA_DIR \
--fig_title $FIG_TITLE \
--png_path $PNG_PATH \
--csv_path $CSV_PATH \
--txt_path $TXT_PATH  
   




