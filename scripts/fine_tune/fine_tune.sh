#!/bin/bash -l

NLP_BARRIERS=/home/gridsan/kspieker/RMG/NLP_barriers
export PYTHONPATH=$NLP_BARRIERS:$PYTHONPATH

# general args
vocab_file=$NLP_BARRIERS/data/mlm_USPTO/original/vocab.txt
train_data=PATH/TO/train.csv
val_data=PATH/TO/val.csv
test_data=PATH/TO/test.csv
best_mlm_model=PATH/TO/final_mlm_checkpoint
targets="dE0"

# TrainingArgs
output_dir='fine_tune'
report_to='wandb'
dataloader_num_workers=6

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

python -u $NLP_BARRIERS/scripts/fine_tune/fine_tune.py \
--vocab_file $vocab_file \
--train_data $train_data \
--val_data $val_data \
--test_data $test_data \
--best_mlm_model $best_mlm_model \
--targets $targets \
--output_dir $output_dir \
--report_to $report_to \
--dataloader_num_workers $dataloader_num_workers \
--fp16 \
--learning_rate $learning_rate \
--lr_scheduler_type $lr_scheduler_type \
--warmup_ratio $warmup_ratio \
--max_grad_norm $max_grad_norm \
--num_train_epochs $num_train_epochs \
--per_device_train_batch_size $per_device_train_batch_size

