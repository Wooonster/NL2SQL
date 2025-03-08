" download model "
import os
import tokenize
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

model_id = "Qwen/Qwen2.5-Coder-1.5B"
save_dir = f"/root/autodl-tmp/NL2SQL/models/{model_id[5:]}/"
os.makedirs(save_dir, exist_ok=True)
# snapshot_download(repo_id=model_id, local_dir=save_dir)

" Load model "
model = AutoModelForCausalLM.from_pretrained(save_dir, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(save_dir)

" Data process"
import pandas as pd
from datasets import Dataset

# system prompt
system_prompt = """You are DeepQuery, a data science expert. Below, you are presented with a database schema, a question and a hint. Your task is to read the schema with annotations of the columns, understand the question and the hint, and generate a valid SQL query to answer the question. You should reason step by step, and include your reasonings between <think> and </think>."""

# 读取 CSV 文件并构造 Dataset
data_dir = '/root/autodl-tmp/NL2SQL/cot-qa.csv'
df = pd.read_csv(data_dir)
dataset = Dataset.from_pandas(df)

def combined_preprocess(batch):
    texts = []
    # 遍历每个样本，构造完整的 prompt 和 completion 文本
    for q, a, t in zip(batch["query"], batch["answer"], batch["thinking_process"]):
        question = str(q)
        answer = str(a)
        thinking = str(t)
        prompt = (
            f"For the question: {question}.\n"
            "Please think step by step, list your thinking process between <think> and </think> and then show the final SQL answer:"
        )
        completion = (
            f"<think>{thinking}</think>\nMy final answer is: ```sql\n{answer}\n```"
        )
        texts.append(prompt + "\n" + completion)
    # 不进行 padding, 也不返回 torch.Tensor, 返回的是列表, 让 collator 统一 pad
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=1024 * 2,
        padding=False,
    )
    print(type(tokenized))
    # print(tokenized.device)
    return tokenized

# 使用 map 时删除原始字段，保证每个样本只包含 tokenized 的输出
processed_dataset = dataset.map(combined_preprocess, batched=True, remove_columns=dataset.column_names)
# print(processed_dataset[0])

" Lora Config "
from peft import LoraConfig, TaskType, get_peft_model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=8,
    lora_alpha=16,  # 8*2
    lora_dropout=0.05,
    bias='none',
    inference_mode=False
)

model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())
model.config.use_cache = False

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
    remove_unused_columns=False,
)

" Swanlab setup "
import swanlab
from swanlab.integration.transformers import SwanLabCallback

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-Coder-1.5B-NL2SQL-SFT",
    experiment_name="Coder-1.5B-NL2SQL-SFT-CoT-BIRD",
    config={
        "model": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B",
        "dataset": "https://modelscope.cn/datasets/ruohuaw/sql-cot",
        "github": "https://github.com/Wooonster/NL2SQL",
        "prompt": "",
        "train_data_number": len(processed_dataset),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt'),
    callbacks=[swanlab_callback],
)

trainer.train()
