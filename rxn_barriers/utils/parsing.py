from argparse import ArgumentParser


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        args: The parsed command-line arguments by key words.
        huggingface_args: Dictionary containing arguments for BertConfig and TrainingArgs
    """
    parser = ArgumentParser()

    parser.add_argument('--vocab_file', type=str,
                        help="Path to the text file containing tokenizer's vocab.")

    parser.add_argument('--train_data', type=str,
                        help='Path to training data.')

    parser.add_argument('--val_data', type=str,
                        help='Path to the validation data.')

    parser.add_argument('--test_data', type=str,
                        help='Path to the testing data.')

    parser.add_argument('--final_mlm_checkpoint', type=str,
                        help='Path to the best MLM checkpoint.')

    parser.add_argument('--targets', nargs='+',
                        help='Name of columns to use as regression targets.')

    # BertConfig arguments
    # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/configuration_bert.py#L72
    bert_config = parser.add_argument_group('BertConfig')

    bert_config.add_argument('--hidden_size', type=int, default=768,
                             help=' Dimensionality of the encoder layers and the pooler layer.')

    bert_config.add_argument('--num_hidden_layers', type=int, default=12,
                             help='Number of hidden layers in the Transformer encoder.')

    bert_config.add_argument('--num_attention_heads', type=int, default=12,
                             help='Number of attention heads for each attention layer in the Transformer encoder.')

    bert_config.add_argument('--intermediate_size', type=int, default=512,
                             help='Dimensionality of the "intermediate" (i.e. feed-forward) layer in the Transformer encoder.')

    bert_config.add_argument('--hidden_act', type=str, default='gelu',
                             choices=['gelu', 'relu', 'silu', 'gelu_new'],
                             help='Non-linear activation function in the encoder and pooler.')

    bert_config.add_argument('--hidden_dropout_prob', type=float, default=0.1,
                             help='Dropout probability for all fully connected layers in the embeddings, encoder, and pooler.')

    bert_config.add_argument('--attention_probs_dropout_prob', type=float, default=0.1,
                             help='Dropout ratio for the attention probabilities.')

    bert_config.add_argument('--max_position_embeddings', type=int, default=512,
                             help='Maximum sequence length that this model might ever be used with.')

    bert_config.add_argument('--type_vocab_size', type=int, default=2,
                             help='Vocabulary size of the token_type_ids passed when calling BertModel.')

    bert_config.add_argument('--initializer_range', type=float, default=0.02,
                             help='Standard deviation of the truncated_normal_initializer for initializing all weight matrices.')

    bert_config.add_argument('--layer_norm_eps', type=float, default=1e-12,
                             help='Epsilon used by the layer normalization layers.')

    # TrainingArgs
    # https://huggingface.co/docs/transformers/v4.20.0/en/main_classes/trainer#transformers.TrainingArguments
    # https://github.com/huggingface/transformers/blob/v4.20.0/src/transformers/training_args.py#L104
    training_args = parser.add_argument_group('TrainingArgs')

    training_args.add_argument('--output_dir', type=str, default='results',
                               help='Output directory where the model predictions and checkpoints will be written.')

    training_args.add_argument('--seed', type=int, default=42,
                               help='Random seed that will be set at the beginning of training.')

    training_args.add_argument('--report_to', type=str, default=None,
                               choices=['azure_ml', 'comet_ml', 'mlflow', 'tensorboard', 'wandb'],
                               help='Where to report the results and logs to.')

    training_args.add_argument('--dataloader_num_workers', type=int, default=0,
                               help='Number of workers for the parallel data loading (0 means sequential).')

    training_args.add_argument('--dataloader_pin_memory', action='store_true', default=False,
                               help='Whether you want to pin memory in data loaders or not. Will default to True.')

    training_args.add_argument('--fp16', action='store_true', default=False,
                               help='Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.')

    # optimzation arguments
    training_args.add_argument('--learning_rate', type=float, default=1e-4,
                               help='Initial learning rate for AdamW optimizer.')

    training_args.add_argument('--lr_scheduler_type', type=str, default='linear',
                               choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'],
                               help='Initial learning rate for AdamW optimizer.')

    training_args.add_argument('--warmup_ratio', type=float, default=0.02,
                               help='Ratio of total training steps used for a linear warmup from 0 to learning_rate.')

    training_args.add_argument('--weight_decay', type=float, default=0.0,
                               help='Weight decay to apply to all layers except all bias and LayerNorm weights in AdamW optimizer.')

    training_args.add_argument('--max_grad_norm', type=float, default=1.0,
                               help='Maximum gradient norm (for gradient clipping).')

    training_args.add_argument('--num_train_epochs', type=float, default=100,
                               help='Total number of training epochs to perform.')

    training_args.add_argument('--per_device_train_batch_size', type=int, default=32,
                               help='Batch size per GPU/TPU core/CPU for training.')

    training_args.add_argument('--per_device_eval_batch_size', type=int, default=None,
                               help='Batch size per GPU/TPU core/CPU for evaluation.')

    # evaluation args
    training_args.add_argument('--evaluation_strategy', type=str, default='epoch',
                               choices=['no', 'steps', 'epoch'],
                               help='Evaluation strategy to adopt during training.')

    training_args.add_argument('--eval_steps', type=int, default=None,
                               help='Number of update steps between two evaluations if evaluation_strategy="steps".')

    training_args.add_argument('--load_best_model_at_end', action='store_true', default=False,
                               help='When set to True, save_strategy must be the same as evaluation_strategy.')

    # saving args
    training_args.add_argument('--save_strategy', type=str, default='epoch',
                               choices=['no', 'steps', 'epoch'],
                               help='Checkpoint strategy to adopt during training.')

    training_args.add_argument('--save_steps', type=int, default=None,
                               help='Number of updates steps before two checkpoint saves if save_strategy="steps".')

    training_args.add_argument('--save_total_limit', type=int, default=5,
                               help='Limits the total amount of checkpoints. Deletes the older checkpoints in output_dir.')

    args = parser.parse_args(command_line_args)

    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.per_device_train_batch_size

    huggingface_args = dict({})
    group_list = ['BertConfig', 'TrainingArgs']
    for group in parser._action_groups:
        if group.title in group_list:
            huggingface_args[group.title] = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}

    return args, huggingface_args
