import torch
import transformers
from transformers import (
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


args, huggingface_args = parse_command_line_arguments()
print('Using arguments...')
for arg in vars(args):
    print(f'{arg}: {getattr(args, arg)}')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# create tokenizer
smi_tokenizer = SmilesTokenizer(vocab_file=args.vocab_file, 
                                # must add this to be able to pad during preprocessing later on
                                model_max_length=args.max_position_embeddings)

bert_config_dict = huggingface_args['BertConfig']
bert_config_dict['vocab_size'] = smi_tokenizer.vocab_size

# set a configuration for our Bert model
config = BertConfig(**bert_config_dict)

# initialize the model from a configuration without pretrained weights
model = BertForMaskedLM(config=config)
print(f'Num parameters: {model.num_parameters():,}')
print(f'Model architecture is:\n{model}')

train_dataset = RxnDatasetMLM(file_path=args.train_data, tokenizer=smi_tokenizer)
test_dataset = RxnDatasetMLM(file_path=args.val_data, tokenizer=smi_tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=smi_tokenizer, mlm=True, mlm_probability=0.15,
)

wandb.init(project="USPTO_MLM_supercloud",
           entity="kspieker",
           mode='offline',
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
