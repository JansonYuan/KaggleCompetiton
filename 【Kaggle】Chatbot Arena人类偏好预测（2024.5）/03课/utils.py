from transformers.optimization import AdamW
from transformers.optimization import Adafactor, get_scheduler
import torch.nn as nn
from torch.optim import lr_scheduler
import torch

def create_deberta_optimizer(args, model):
    """
    Setup the optimizer.
    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.

    MODIFIED VERSION:
    * added support for differential learning rates per layer

    reference: https://github.com/huggingface/transformers/blob/05fa1a7ac17bb7aa07b9e0c1e138ecb31a28bbfe/src/transformers/trainer.py#L804
    """

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    ### ADDED
    if args.discriminative_learning_rate:

        num_layers = model.config.num_hidden_layers

        learning_rate_powers = range(0, num_layers, num_layers // args.discriminative_learning_rate_num_groups)
        layer_wise_learning_rates = [
            pow(args.discriminative_learning_rate_decay_rate, power) * args.learning_rate
            for power in learning_rate_powers
            for _ in range(num_layers // args.discriminative_learning_rate_num_groups)
        ]
        layer_wise_learning_rates = layer_wise_learning_rates[::-1]
        print('Layer-wise learning rates:', layer_wise_learning_rates)

        # group embedding paramters from the transformer encoder
        embedding_layer = model.base_model.embeddings
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in embedding_layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": pow(args.discriminative_learning_rate_decay_rate, num_layers) * args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in embedding_layer.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": pow(args.discriminative_learning_rate_decay_rate, num_layers) * args.learning_rate,
                "weight_decay": 0.0,
            },
        ]

        # group encoding paramters from the transformer encoder
        encoding_layers = [layer for layer in model.base_model.encoder.layer]
        for i, layer in enumerate(encoding_layers):
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "lr": layer_wise_learning_rates[i],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "lr": layer_wise_learning_rates[i],
                    "weight_decay": 0.0,
                },
            ]
        print(
            f"Detected unattached modules in model.encoder: {[n for n, p in model.base_model.encoder.named_parameters() if not n.startswith('layer')]}")
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.base_model.encoder.named_parameters() if
                           not n.startswith('layer') and not any(nd in n for nd in no_decay)],
                "lr": layer_wise_learning_rates[-1],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.base_model.encoder.named_parameters() if
                           not n.startswith('layer') and any(nd in n for nd in no_decay)],
                "lr": layer_wise_learning_rates[-1],
                "weight_decay": 0.0,
            },
        ]

        # group paramters from the task specific head
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.named_parameters() if
                           'deberta' not in n and not any(nd in n for nd in no_decay)],
                "lr": args.head_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           'deberta' not in n and any(nd in n for nd in no_decay)],
                "lr": args.head_lr,
                "weight_decay": 0.0,
            },
        ]
    ### END ADDED
    else:
        # group paramters for the entire network
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": args.learning_rate,
                "weight_decay": 0.0,
            },
        ]

    if args.optim == "adafactor":
        print("Using Adafactor")
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}

    else:
        print("Using adam")
        if args.adam_optim_bits == 8:
            optimizer_cls = bnb.optim.AdamW
            optimizer_kwargs = {
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
                "optim_bits": args.adam_optim_bits,
            }
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
            }

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    # make sure to optimize nn.Embedding with 32-bit AdamW
    # reference: https://github.com/huggingface/transformers/issues/14819#issuecomment-1003445038
    if optimizer_cls.__name__ == "Adam8bit":
        manager = bnb.optim.GlobalOptimManager.get_instance()
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                manager.register_module_override(module, "weight", {"optim_bits": 32})

    return optimizer



def create_custom_deberta_optimizer(args, model):
    """
    Setup the optimizer.
    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.

    MODIFIED VERSION:
    * added support for differential learning rates per layer

    reference: https://github.com/huggingface/transformers/blob/05fa1a7ac17bb7aa07b9e0c1e138ecb31a28bbfe/src/transformers/trainer.py#L804
    """

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    ### ADDED
    if args.discriminative_learning_rate:

        num_layers = model.config.num_hidden_layers

        learning_rate_powers = range(0, num_layers, num_layers // args.discriminative_learning_rate_num_groups)
        layer_wise_learning_rates = [
            pow(args.discriminative_learning_rate_decay_rate, power) * args.learning_rate
            for power in learning_rate_powers
            for _ in range(num_layers // args.discriminative_learning_rate_num_groups)
        ]
        layer_wise_learning_rates = layer_wise_learning_rates[::-1]
        print('Layer-wise learning rates:', layer_wise_learning_rates)

        # group embedding paramters from the transformer encoder
        embedding_layer = model.bert_model.base_model.embeddings
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in embedding_layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": pow(args.discriminative_learning_rate_decay_rate, num_layers) * args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in embedding_layer.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": pow(args.discriminative_learning_rate_decay_rate, num_layers) * args.learning_rate,
                "weight_decay": 0.0,
            },
        ]

        # group encoding paramters from the transformer encoder
        encoding_layers = [layer for layer in model.bert_model.base_model.encoder.layer]
        for i, layer in enumerate(encoding_layers):
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "lr": layer_wise_learning_rates[i],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "lr": layer_wise_learning_rates[i],
                    "weight_decay": 0.0,
                },
            ]
        print(
            f"Detected unattached modules in model.encoder: {[n for n, p in model.bert_model.base_model.encoder.named_parameters() if not n.startswith('layer')]}")
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.bert_model.base_model.encoder.named_parameters() if
                           not n.startswith('layer') and not any(nd in n for nd in no_decay)],
                "lr": layer_wise_learning_rates[-1],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.bert_model.base_model.encoder.named_parameters() if
                           not n.startswith('layer') and any(nd in n for nd in no_decay)],
                "lr": layer_wise_learning_rates[-1],
                "weight_decay": 0.0,
            },
        ]

        # group paramters from the task specific head
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model.named_parameters() if
                           'bert_model' not in n and not any(nd in n for nd in no_decay)],
                "lr": args.head_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           'bert_model' not in n and any(nd in n for nd in no_decay)],
                "lr": args.head_lr,
                "weight_decay": 0.0,
            },
        ]
    ### END ADDED
    else:
        # group paramters for the entire network
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": args.learning_rate,
                "weight_decay": 0.0,
            },
        ]

    if args.optim == "adafactor":
        print("Using Adafactor")
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}

    else:
        print("Using adam")
        if args.adam_optim_bits == 8:
            optimizer_cls = bnb.optim.AdamW
            optimizer_kwargs = {
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
                "optim_bits": args.adam_optim_bits,
            }
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
            }

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    # make sure to optimize nn.Embedding with 32-bit AdamW
    # reference: https://github.com/huggingface/transformers/issues/14819#issuecomment-1003445038
    if optimizer_cls.__name__ == "Adam8bit":
        manager = bnb.optim.GlobalOptimManager.get_instance()
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                manager.register_module_override(module, "weight", {"optim_bits": 32})

    return optimizer

def create_scheduler(args, model, num_training_steps: int, optimizer: torch.optim.Optimizer,
                     lr_scheduler_type='CosineAnnealingLR'):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.
    Args:
      num_training_steps (int): The number of training steps to do.
    """

    if lr_scheduler_type == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0)
    elif lr_scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, eta_min=0)
    else:
        scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )

    return scheduler
