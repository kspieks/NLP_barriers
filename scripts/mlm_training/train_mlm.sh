#!/bin/bash -l

NLP_BARRIERS=/home/gridsan/kspieker/RMG/NLP_barriers
export PYTHONPATH=$NLP_BARRIERS:$PYTHONPATH

# general args
vocab_file=$NLP_BARRIERS/data/mlm_USPTO/original/vocab.txt
mlm_train_path=$NLP_BARRIERS/data/mlm_USPTO/with_Hs/mlm_train_file_Hs.txt
mlm_eval_path=$NLP_BARRIERS/data/mlm_USPTO/with_Hs/mlm_eval_file_Hs.txt

# BertConfig arguments
hidden_size=512
num_hidden_layers=6
num_attention_heads=8
intermediate_size=512
hidden_act='gelu'
hidden_dropout_prob=0.1
attention_probs_dropout_prob=0.1
max_position_embeddings=512

# TrainingArgs
output_dir='MLM_test'
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

# -u: Force stdin, stdout and stderr to be totally unbuffered
# On systems where it matters, also put stdin, stdout and stderr in binary mode
python -u $NLP_BARRIERS/mlm_training.py \
--vocab_file $vocab_file \
--mlm_train_path $mlm_train_path \
--mlm_eval_path $mlm_eval_path \
--hidden_size $hidden_size \
--num_hidden_layers $num_hidden_layers \
--num_attention_heads $num_attention_heads \
--intermediate_size $intermediate_size \
--hidden_act $hidden_act \
--hidden_dropout_prob $hidden_dropout_prob \
--attention_probs_dropout_prob $attention_probs_dropout_prob \
--max_position_embeddings $max_position_embeddings \
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

