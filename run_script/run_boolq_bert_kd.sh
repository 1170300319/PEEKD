export TASK_NAME=superglue
export DATASET_NAME=boolq
export CUDA_VISIBLE_DEVICES=0,1,2,3

bs=8
lr=5e-3
dropout=0.1
psl=40
psl_t=40
epoch=100

python3 run.py \
  --model_name_or_path bert-base-cased \
  --teacher_name_or_path checkpoints/$DATASET_NAME-bert_moe_teacher/ \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --moe True \
  --kd True \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --pre_seq_len_t $psl_t \
  --output_dir checkpoints/$DATASET_NAME-bert_moe_kd/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
