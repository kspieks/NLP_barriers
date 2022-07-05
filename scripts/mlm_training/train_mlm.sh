#!/bin/bash -l

NLP_BARRIERS=/home/gridsan/kspieker/RMG/NLP_barriers
export PYTHONPATH=$NLP_BARRIERS:$PYTHONPATH

# general args
vocab_file=$NLP_BARRIERS/data/mlm_RMG/vocab.txt
train_data=$NLP_BARRIERS/data/mlm_RMG/mlm_RMG_train.txt
val_data=$NLP_BARRIERS/data/mlm_RMG/mlm_RMG_val.txt

# Config arguments
config_json=bert_config.json

# TrainingArgs
output_dir='MLM_test'
report_to='wandb'
dataloader_num_workers=6

# dataloader_pin_memory
# fp16

learning_rate=1e-4
lr_scheduler_type='cosine'
warmup_ratio=0.02
max_grad_norm=5.0
num_train_epochs=10
per_device_train_batch_size=32

evaluation_strategy='epoch'
save_strategy='epoch'
save_total_limit=5

source activate huggingface
which python
python -c "import torch;print(torch.cuda.device_count());print(torch.cuda.is_available())"

# -u: Force stdin, stdout and stderr to be totally unbuffered
# On systems where it matters, also put stdin, stdout and stderr in binary mode
python -u $NLP_BARRIERS/scripts/mlm_training/mlm_training.py \
--vocab_file $vocab_file \
--train_data $train_data \
--val_data $val_data \
--config_json $config_json \
--output_dir $output_dir \
--report_to $report_to \
--dataloader_num_workers $dataloader_num_workers \
--fp16 \
--learning_rate $learning_rate \
--lr_scheduler_type $lr_scheduler_type \
--warmup_ratio $warmup_ratio \
--max_grad_norm $max_grad_norm \
--num_train_epochs $num_train_epochs \
--per_device_train_batch_size $per_device_train_batch_size \
--evaluation_strategy $evaluation_strategy \
--load_best_model_at_end $load_best_model_at_end \
--save_strategy $save_strategy \
--save_total_limit $save_total_limit

