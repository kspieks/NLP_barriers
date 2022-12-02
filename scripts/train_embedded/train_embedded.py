import json
import os
import torch
# torch.backends.cudnn.benchmark = True
from transformers import (
    AlbertConfig,
    AlbertForSequenceClassification,
    BertConfig,
    BertForSequenceClassification, 
    TrainingArguments,
    RobertaModel,
)

import wandb

from rxn_barriers.data import RxnDatasetEmbeddingsRegression, TorchStandardScaler
from rxn_barriers.tokenization import SmilesTokenizer
from rxn_barriers.utils.parsing import parse_command_line_arguments
from rxn_barriers.utils.nn_utils import CustomTrainer

EMBEDDER_CLASSES = {
    'roberta': (RobertaModel, SmilesTokenizer),
}

MODEL_CLASSES = {
            'albert': (AlbertConfig, AlbertForSequenceClassification),
            'bert': (BertConfig, BertForSequenceClassification),
}

args, huggingface_args = parse_command_line_arguments()

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# get embedder and tokenizer classes
embedder_class, tokenizer_class = EMBEDDER_CLASSES[args.embedder_type.lower()]

# create tokenizer
with open(args.embedder_config_json, 'r') as f:
    embedder_config_dict = json.load(f)
smi_tokenizer = tokenizer_class(
    vocab_file=args.vocab_file, 
    model_max_length=embedder_config_dict['max_position_embeddings']-2,
)

print('Using arguments...')
for arg in vars(args):
    print(f'{arg}: {getattr(args, arg)}')

# instantiate pretrained embedder
embedder = embedder_class.from_pretrained(
    args.embedder_path,
)
print(f'Num embedder parameters: {embedder.num_parameters():,}')
print(f'Embedder architecture is:\n{embedder}')

# create datasets
train_dataset = RxnDatasetEmbeddingsRegression(data_path=args.train_data, tokenizer=smi_tokenizer, targets=args.targets, embedder=embedder)
val_dataset = RxnDatasetEmbeddingsRegression(data_path=args.val_data, tokenizer=smi_tokenizer, targets=args.targets, embedder=embedder)
test_dataset = RxnDatasetEmbeddingsRegression(data_path=args.test_data, tokenizer=smi_tokenizer, targets=args.targets, embedder=embedder)

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

test_dataset.labels = scaler.transform(torch.tensor(test_dataset.labels, dtype=torch.float))
print(f'Test scaled mean: {test_dataset.mean} +- std {test_dataset.std} kcal/mol')

model_config_class, model_class = MODEL_CLASSES[args.model_type.lower()]

# instantiate model
with open(args.config_json, 'r') as f:
    model_config_dict = json.load(f)

print(f'Using model config:\n{model_config_dict}')

model_config = model_config_class.from_dict(model_config_dict)
model = model_class(model_config)

print(f'Num model parameters: {model.num_parameters():,}')
print(f'Model architecture is:\n{model}')

# wandb.init(project=args.wandb_project,
#            entity=args.wandb_entity,
        #    mode=args.wandb_mode,
        #    config=args,
        #    )

os.environ["WANDB_DISABLED"] = "true"

# set up trainer
training_args_dict = huggingface_args['TrainingArgs']
training_args = TrainingArguments(**training_args_dict)

trainer = CustomTrainer(scaler=scaler,
                        targets=args.targets,
                        mode='val',
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        )
trainer.train()

# get validation predictions
output = trainer.evaluate(eval_dataset=val_dataset,
                          metric_key_prefix='val',
                         )
print('Output from validation')
print(output)

# get testing predictions
output = trainer.evaluate(eval_dataset=test_dataset,
                          metric_key_prefix='test',)
print('Output from testing')
print(output)
