export TASK_NAME=superglue
export DATASET_NAME=rte
export CUDA_VISIBLE_DEVICES=0,1,2,3

bs=8
lr=1e-2
dropout=0.1
psl=128
psl_t=128
epoch=60

python3 run.py \
  --model_name_or_path bert-base-cased \
  --teacher_name_or_path checkpoints/$DATASET_NAME-bert_moe_teacher/ \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --kd True \
  --moe True \
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
