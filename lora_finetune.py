from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, PeftModel, PeftConfig, LoraConfig, get_peft_model, TaskType
import torch
import wandb
import os

os.environ["WANDB_PROJECT"] = "gemma"wandb.init()

# Import the pretrained model 
from HuggingFace without quantizationmodel_name = "state-spaces/mamba-2.8b-hf"
from accelerate import PartialState

device_string = PartialState().process_indexmodel = AutoModelForCausalLM.from_pretrained(    model_name,    device_map="auto")
print(model)# Initialize LoRa configconfig = LoraConfig(    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128    

# target_modules = ["in_proj", "x_proj", "dt_proj", "out_proj"],    
target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],    
lora_alpha=16,    
lora_dropout=0,  

# Supports any, but = 0 is optimized    task_type="CAUSAL_LM",    bias="none",  # Supports any, but = "none" is optimized)
from unsloth.chat_templates import get_chat_template
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
tokenizer = get_chat_template(    
  tokenizer,    
  chat_template="chatml",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth    
  mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style    
  map_eos_token=True,  # Maps <|im_end|> to </s> instead
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = load_dataset("Open-Orca/SlimOrca-Dedup", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size4,
    logging_dir='./logs',
    logging_steps=10,    learning_rate=2e-3,    report_to="wandb",)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=training_args,
    peft_config=config,
)

# Train model
trainer.train()

wandb.finish()
