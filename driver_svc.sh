#!/bin/sh
MODEL_NAME='svc'
source ./setup.sh $MODEL_NAME



python src/svc.py \
--start_gamma -2 --end_gamma 3 --step_gamma 1 \
--start_c -2 --end_c 3 --step_c 1 \
--model_name $MODEL_NAME \
--output_dir $OUTPUT_DIR \
--data_dir $DATA_DIR \
--fig_title $FIG_TITLE \
--png_path $PNG_PATH \
--csv_path $CSV_PATH \
--txt_path $TXT_PATH  
   



