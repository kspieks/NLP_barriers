import json

import torch
import transformers
from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
    BertConfig, 
    BertForMaskedLM, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments,
)
import wandb
print(f'Using transformers version: {transformers.__version__}')

from rxn_barriers.data import RxnDatasetMLM
from rxn_barriers.tokenization import SmilesTokenizer
from rxn_barriers.utils.parsing import parse_command_line_arguments


MODEL_CLASSES = {
            'albert': (AlbertConfig, AlbertForMaskedLM, SmilesTokenizer),
            'bert': (BertConfig, BertForMaskedLM, SmilesTokenizer),
}

args, huggingface_args = parse_command_line_arguments()

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type.lower()]

# create tokenizer
with open(args.config_json, 'r') as f:
    config_dict = json.load(f)
smi_tokenizer = tokenizer_class(vocab_file=args.vocab_file, 
                                # must add this to be able to pad during preprocessing later on
                                model_max_length=config_dict['max_position_embeddings'])

# set model configuration
config_dict['vocab_size'] = smi_tokenizer.vocab_size
for key, value in config_dict.items():
    setattr(args, key, value)
config = config_class(**config_dict)

print('Using arguments...')
for arg in vars(args):
    print(f'{arg}: {getattr(args, arg)}')

# initialize the model from a configuration without pretrained weights
model = model_class(config=config)
print(f'Num parameters: {model.num_parameters():,}')
print(f'Model architecture is:\n{model}')

train_dataset = RxnDatasetMLM(file_path=args.train_data, tokenizer=smi_tokenizer)
test_dataset = RxnDatasetMLM(file_path=args.val_data, tokenizer=smi_tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=smi_tokenizer, mlm=True, mlm_probability=0.15,
)

wandb.init(project=args.wandb_project,
           entity=args.wandb_entity,
           mode=args.mode,
           config=args,
           )

training_args_dict = huggingface_args['TrainingArgs']
training_args = TrainingArguments(**training_args_dict)

trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=train_dataset,
                  eval_dataset=test_dataset,
                  tokenizer=smi_tokenizer,
                  )

trainer.train()
