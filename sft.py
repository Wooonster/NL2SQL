" download model "
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-Coder-1.5B"
save_dir = f"/root/autodl-tmp/NL2SQL/models/{model_id[5:]}/"
os.makedirs(save_dir, exist_ok=True)
# snapshot_download(repo_id=model_id, local_dir=save_dir)

" Load model "
model = AutoModelForCausalLM.from_pretrained(save_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(save_dir, device_map="auto")

" Data process"
import pandas as pd
# system prompt
system_prompt = """You are DeepQuery, a data science expert. Below, you are presented with a database schema, a question and a hint.Your task is to read the schema with annotations of the columns, understand the question and the hint, and generate a valid SQL query to answer the question. You should reason step by step, and includes your reasonings between <think> and </think>."""

# SFT dataset
# CoT enhanced BIRD
data_dir = '/root/autodl-tmp/NL2SQL/sql-cot/cot-qa.csv'

df = pd.read_csv(data_dir)
# print(df.head(3))

# query, answer, thinking = df["query"], df["answer"], df["thinking_process"]
def process_data(row):
    question, answer, thinking = row["query"], row["answer"], row["thinking_process"]
    prompt = f"For the question: {question}.\nPlease think step by step, list your thinking process between <think> and </think> and then show the final SQL answer:"
    completion = f"<think>{thinking}</think>\nMy final answer is: ```sql\n{answer}\n```"
    return {"prompt": prompt, "completion": completion}

training_data = df.apply(process_data, axis=1).tolist()
print(f"training data size: {len(training_data)}")  # 9399

" Tokenize and Build dataset "
def tokenize_func(example):
    text = example["prompt"] + '\n' + example["completion"]
    return tokenizer(text, truncation=True, max_length=1024*2)

tokenized_data = [tokenize_func(example) for example in training_data]

import torch
from torch.utils.data import Dataset

class NL2SQLDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

training_dataset = NL2SQLDataset(tokenized_data)

" Lora Config "
from peft import LoraConfig, TaskType, get_peft_model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=8*2,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj',],
    bias='none',
    inference_mode=False
)

model = get_peft_model(model, lora_config)

" Training Config "
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output/sft/",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

" Swanlab setup "
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen2.5-Coder-1.5B-NL2SQL-SFT",
    experiment_name="Coder-1.5B-NL2SQL-SFT-CoT-BIRD",
    config={
        "model": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B",
        "dataset": "https://modelscope.cn/datasets/ruohuaw/sql-cot",
        "github": "https://github.com/Wooonster/NL2SQL",
        "prompt": "https://github.com/Wooonster/NL2SQL",
        "train_data_number": len(training_data),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    tokenizer=tokenizer,
    callbacks=[swanlab_callback],
)

trainer.train()