#!/bin/sh

MODEL_NAME='mlp'
source ./setup.sh $MODEL_NAME



python src/mlp.py \
--start_nhidden 3 --end_nhidden 21 --step_nhidden 1 \
--batch_size 64 \
--epochs 100 \
--learning_rate 3e-2 \
--model_name $MODEL_NAME \
--output_dir $OUTPUT_DIR \
--rbf $RBF \
--data_dir $DATA_DIR \
--fig_title $FIG_TITLE \
--png_path $PNG_PATH \
--csv_path $CSV_PATH  



   



