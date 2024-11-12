import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # Added this import
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
import logging
import wandb
from datetime import timedelta
from datetime import datetime
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
)
from tqdm.auto import tqdm
import torch.nn as nn
from typing import Dict, Union, Any
from transformers import get_cosine_schedule_with_warmup

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples):
        batch = {
            key: [example[key] for example in examples]
            for key in examples[0].keys()
        }

        input_ids = torch.tensor(batch["input_ids"])
        attention_mask = torch.tensor(batch["attention_mask"])
        labels = torch.tensor(batch["labels"])

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            labels = labels.unsqueeze(0)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand_as(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids
        }

class CustomTrainer(Trainer):
    def _wrap_model(self, model, training=True):
        if hasattr(model, "module"):
            # Model is already DDP wrapped, access base model
            base_model = model.module
            base_model.gradient_checkpointing_enable()
        else:
            # Enable gradient checkpointing before DDP wrap
            model.gradient_checkpointing_enable()
        
        # Now wrap with DDP
        if self.args.local_rank != -1:
            wrapped_model = super()._wrap_model(model, training)
            return wrapped_model
        return model

    def _inner_training_loop(self, *args, **kwargs):
        # Skip the gradient checkpointing setup in the parent class
        args_dict = kwargs.get("args", self.args)
        
        # Store original value
        original_gradient_checkpointing = args_dict.gradient_checkpointing
        
        # Temporarily disable gradient checkpointing flag since we handle it ourselves
        args_dict.gradient_checkpointing = False
        
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            # Restore original value
            args_dict.gradient_checkpointing = original_gradient_checkpointing


def setup_wandb(local_rank):
    if local_rank in [-1, 0]:
        run = wandb.init(
            project="llama-finetune",
            name=f"fsdp-run-{wandb.util.generate_id()}",
            config={
                "learning_rate": 1e-5,
                "epochs": 3,
                "batch_size": 8,
                "model": "llama-3.2-2b",
                "weight_decay": 0.01,
            }
        )
        return run
    return None

def setup_accelerator():
    is_gpu_available = torch.cuda.is_available()
    
    if is_gpu_available:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
    else:
        mixed_precision_policy = None

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        sharding_strategy="NO_SHARD",  # Changed from FULL_SHARD
        mixed_precision_policy=mixed_precision_policy,
        backward_prefetch="BACKWARD_POST",
    )

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))

    accelerator = Accelerator(
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[kwargs],
        gradient_accumulation_steps=4,
        log_with="wandb",
        mixed_precision="fp16" if is_gpu_available else None,
        device_placement=True,
        dispatch_batches=True  # Added this
    )
    
    return accelerator, is_gpu_available

def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )

def main():
    logger = setup_logging()
    logger.info("Starting training")
    
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_rank = int(os.environ.get("RANK", "0"))
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )

        # Load dataset
        df = pd.read_csv('metaqa.csv')
        train_dataset = Dataset.from_pandas(df[['sentence']])
        
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B', trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        def tokenize_function(examples):
            result = tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors=None,
                return_attention_mask=True
            )
            result["labels"] = result["input_ids"].copy()
            return result
            
        tokenized_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=32,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing"
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join('/scratch/arjun.dosajh/llama_finetune', f"run_{timestamp}")
        if local_rank <= 0:
            os.makedirs(save_dir, exist_ok=True)

        # Modified training arguments to handle mixed precision properly
        training_args = TrainingArguments(
            output_dir=save_dir,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            save_only_model=True,
            learning_rate=1e-5,
            num_train_epochs=300,
            weight_decay=0.01,
            fp16=False,  # Disable fp16 training
            bf16=True,  # Use bfloat16 instead
            per_device_train_batch_size=8,
            gradient_accumulation_steps=16,
            max_grad_norm=1.0,
            local_rank=local_rank,
            ddp_backend="nccl" if torch.cuda.is_available() else "gloo",
            gradient_checkpointing=True,
            logging_steps=1,
            remove_unused_columns=False,
            report_to=["wandb"] if local_rank <= 0 else [],
            warmup_steps=500,
            lr_scheduler_type="cosine_with_restarts",
            warmup_ratio=0.1,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            dataloader_pin_memory=False,
        )

        # Load model with bfloat16 dtype
        model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Llama-3.2-3B',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            use_cache=False
        )
        
        if torch.cuda.is_available():
            model = model.to(f'cuda:{local_rank}')
            
        model.gradient_checkpointing_enable()
        
        if dist.is_initialized():
            model = DDP(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,
                broadcast_buffers=False
            )

        # Use custom trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=CustomDataCollator(tokenizer),
            train_dataset=tokenized_dataset,
        )

        logger.info("Starting training loop...")
        trainer.train()
            
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise e
    finally:
        if wandb.run is not None and local_rank <= 0:
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info("Training completed")

if __name__ == "__main__":
    main()
