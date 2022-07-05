import os

import torch
import transformers
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments,
)
import wandb

from rxn_barriers.data import RxnDatasetRegression, TorchStandardScaler
from rxn_barriers.tokenization import SmilesTokenizer
from rxn_barriers.utils.parsing import parse_command_line_arguments
from rxn_barriers.utils.nn_utils import CustomTrainer


args, huggingface_args = parse_command_line_arguments()
print('Using arguments...')
for arg in vars(args):
    print(f'{arg}: {getattr(args, arg)}')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# create tokenizer
smi_tokenizer = SmilesTokenizer(vocab_file=args.vocab_file, 
                                model_max_length=args.max_position_embeddings,
                                )

# instantiate pretrained model
model = BertForSequenceClassification.from_pretrained(
    args.final_mlm_checkpoint,
    problem_type='regression',
    num_labels=1,
)
print(f'Num parameters: {model.num_parameters():,}')
print(f'Model architecture is:\n{model}')

# create datasets
train_dataset = RxnDatasetRegression(data_path=args.train_data, tokenizer=smi_tokenizer, targets=args.targets)
val_dataset = RxnDatasetRegression(data_path=args.val_data, tokenizer=smi_tokenizer, targets=args.targets)
test_dataset = RxnDatasetRegression(data_path=args.test_data, tokenizer=smi_tokenizer, targets=args.targets)

scaler = TorchStandardScaler()
scaler.fit(torch.tensor(train_dataset.labels, dtype=torch.float))

print(f'\nTraining mean +- 1 std: {train_dataset.mean} +- {train_dataset.std} kcal/mol')
print(f'Validation mean +- 1 std: {val_dataset.mean} +- {val_dataset.std} kcal/mol')
print(f'Testing mean +- 1 std: {test_dataset.mean} +- {test_dataset.std} kcal/mol\n')

train_dataset.labels = scaler.transform(torch.tensor(train_dataset.labels, dtype=torch.float))
print(f'Training after z-score...')
print(f'mean: {train_dataset.mean} +- std {train_dataset.std} kcal/mol')

val_dataset.labels = scaler.transform(torch.tensor(val_dataset.labels, dtype=torch.float))
print(f'Val scaled mean: {val_dataset.mean} +- std {val_dataset.std} kcal/mol')

wandb.init(project="fine_tune_supercloud",
           entity="kspieker",
           mode='offline',
           )

training_args_dict = huggingface_args['TrainingArgs']
training_args = TrainingArguments(**training_args_dict)

trainer = CustomTrainer(scaler=scaler,
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        tokenizer=smi_tokenizer,
                        )

trainer.train()
