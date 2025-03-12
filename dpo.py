import os
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerArguments


class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(data_path, "r", encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt, chosen, rejected = item["messages"][0]["content"], item["chosen"]["content"], item["rejected"]["content"]

        # tokenize
        prompt = self.tokenizer(prompt, truncation=True, padding=True, max_length=self.max_seq_len, return_tensors=True)
        chosen = self.tokenizer(chosen, truncation=True, padding=True, max_length=self.max_seq_len, return_tensors=True)
        rejected = self.tokenizer(rejected, truncation=True, padding=True, max_length=self.max_seq_len, return_tensors=True)

        print("tokenzied prompt/chosen/rejected shape: ", prompt.shape)

        # remove batch dim
        prompt_ids, prompt_masks = prompt["input_ids"].squeeze(0),  prompt["attention_mask"].squeeze(0)
        chosen_ids, chosen_masks = chosen["input_ids"].squeeze(0),  chosen["attention_mask"].squeeze(0)
        rejected_ids, rejected_masks = rejected["input_ids"].squeeze(0),  rejected["attention_mask"].squeeze(0)

        # concatenate with prompt
        chosen_input_ids = torch.cat([prompt_ids, chosen_ids], dim=0)
        chosen_attention_mask = torch.cat([prompt_masks, chosen_masks], dim=0)

        rejected_input_ids = torch.cat([prompt_ids, rejected_ids], dim=0)
        rejected_attention_mask = torch.cat([prompt_masks, rejected_masks], dim=0)

        # record prompt token amount
        prompt_len = prompt_ids.shape[0]

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "prompt_len": prompt_len
        }


class DPODataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        chosen_input_ids = [f["chosen_input_ids"] for f in features]
        chosen_attention_mask = [f["chosen_attention_mask"] for f in features]
        rejected_input_ids = [f["rejected_input_ids"] for f in features]
        rejected_attention_mask = [f["rejected_attention_mask"] for f in features]
        prompt_len = [f["prompt_len"] for f in features]

        # padding
        batch_chosen = self.tokenizer.pad(
            {"input_ids": chosen_input_ids, "attention_mask": chosen_attention_mask},
            return_tensors='pt'
        )

        batch_rejected = self.tokenizer.pad(
            {"input_ids": rejected_input_ids, "attention_mask": rejected_attention_mask},
            return_tensors='pt'
        )

        return {
            "chosen_input_ids": batch_chosen["input_ids"],
            "chosen_attention_mask": batch_chosen["attention_mask"],
            "rejected_input_ids": batch_rejected["input_ids"],
            "rejected_attention_mask": batch_rejected["attention_mask"],
            "prompt_len": torch.tensor(prompt_len)
        }


def logits_to_probs(logits, labels):
    '''
    将 logits 转换为 概率
    args:
        logits: -> (batch_size, seq_len, vocab_size)
        labels: -> (batch_size, seq_len)

    returns:
        prob: -> (batch_size, seq_len)
    '''
    print(f"logits.size() = {logits.size()}, labels.size() = {labels.size()}")
    # 对 logits 进行 log softmax 操作，得到 log 概率
    log_probs = F.log_softmax(logits, dim=2)
    # 使用 labels 作为索引，从 log_probs 中提取对应的 log 概率，并去掉最后一维
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return probs


def mask_logits(logits, labels):
    new_logits = []
    for logit, label in zip(logits, labels):
        new_logits.append(logit[label != 0].sum().unsqueeze(0))
    return new_logits


def dpo_loss(ref_probs, probs, beta):
    def split_probs(probs):
        len_chosen = probs.size(0) // 2
        chosen = probs[:len_chosen]
        rejected = probs[len_chosen:]
        return chosen, rejected
    
    ref_chosen_probs, ref_reject_probs = split_probs(ref_probs)
    chosen_probs, reject_probs = split_probs(probs)
    loss = -F.logsigmoid(beta * ((chosen_probs - reject_probs) - (ref_chosen_probs - ref_reject_probs)))
    return loss.mean()


class DPOTrainer(Trainer):
    def __init__(self, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def compute_loss(self, inputs, model, ref_model, return_outputs=False):
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        prompt_length = inputs["prompt_len"]  # (batch_size,)

        batch_size = prompt_length.size(0)

        # model logits
        chosen_outputs = model(chosen_input_ids, attention_mask=chosen_attention_mask)
        rejected_outputs = model(rejected_input_ids, attention_mask=rejected_attention_mask)

        # 提取 logits
        chosen_logits = chosen_outputs.logits
        rejected_logits = rejected_outputs.logits

        # 计算每个 token 的 log 概率
        chosen_log_probs = logits_to_probs(
            chosen_logits[:, :-1, :],  # 去掉最后一个 token 
            chosen_input_ids[:, 1:]    # labels 由 input_ids 的右移构成 (predict-next-token)
        )
        rejected_log_probs = logits_to_probs(rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])

        # 对于每个样本，仅累加回答部分的 log 概率
        model_chosen_sums = []
        model_rejected_sums = []
        for i in range(batch_size):
            start = prompt_len[i].item() - 1  # 从 prompt 长度 - 1 处开始
            model_chosen_sums.append(chosen_log_probs[i, start:].sum())
            model_rejected_sums.append(rejected_log_probs[i, start:].sum())
        
        model_chosen_sums = torch.stack(model_chosen_sums)
        model_rejected_sums = torch.stack(model_rejected_sums)
        # 将 chosen 与 rejected 的累计概率拼接，形成一个 shape 为 (2 * batch_size,) 的 tensor
        model_probs = torch.cat([model_chosen_sums, model_rejected_sums], dim=0)

        # ref model logits
        with torch.no_grad():
            ref_chosen_outputs = ref_model(chosen_input_ids, attention_mask=chosen_attention_mask)
            ref_rejected_outputs = ref_model(rejected_input_ids, attention_mask=rejected_attention_mask)

            ref_chosen_logits = ref_chosen_outputs.logits
            ref_rejected_logits = ref_rejected_outputs.logits

            ref_chosen_log_probs = logits_to_probs(ref_chosen_logits[:, :-1, :], chosen_input_ids[:, 1:])
            ref_rejected_log_probs = logits_to_probs(ref_rejected_logits[:, :-1, :], rejected_input_ids[:, 1:])

            ref_model_chosen_sums = []
            ref_model_rejected_sums = []
            for i in range(batch_size):
                start = prompt_len[i].item() - 1
                ref_model_chosen_sums.append(ref_chosen_log_probs[i, start:].sum())
                ref_model_rejected_sums.append(ref_rejected_log_probs[i, start:].sum())
            
            ref_model_chosen_sums = torch.stack(ref_model_chosen_sums)
            ref_model_rejected_sums = torch.stack(ref_model_rejected_sums)
            ref_model_probs = torch.cat([ref_model_chosen_sums, ref_model_rejected_sums], dim=0)

        # loss
        loss = dpo_loss(ref_model_probs, model_probs, self.beta)

        if return_outputs:
            return loss, chosen_outputs
        return loss


if __name__ == '__main__':
    save_dir = 'output/sft/checkpoint-1174/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        save_dir, device_map='auto',
        trust_remote_code=True
    )
    model.to(device)

    # load reference model
    model_ref = AutoModelForCausalLM.from_pretrained(
        save_dir, device_map='auto',
        trust_remote_code=True
    )
    # set to eval, prevent gradient calculation
    model_ref.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained('models/Qwen2.5-Coder-1.5B')

    # create dataset
    data_dir = './data/preference_data.json'
    dataset = DPODataset(data_dir, tokenizer, max_seq_len=1024)
    data_collator = DPODataCollator(tokenizer)

    # training args
    args = TrainingArguments(
        output_dir = './output/dpo/',
        num_train_epochs=3,
        do_train=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        logging_steps=10,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        save_steps=100,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )
    
    " Swanlab setup "
    import swanlab
    from swanlab.integration.transformers import SwanLabCallback

    swanlab_callback = SwanLabCallback(
        project="Qwen2.5-Coder-1.5B-NL2SQL-DPO",
        experiment_name="Coder-1.5B-NL2SQL-DPO-preference",
        config={
            "model": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B",
            "dataset": "https://github.com/RUCKBReasoning/DPO_Text2SQL",
            "github": "https://github.com/Wooonster/NL2SQL",
            "prompt": "",
            "train_data_number": len(dataset),
            # "lora_rank": 8,
            # "lora_alpha": 16,
            # "lora_dropout": 0.05,
        },
    )

    
    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=args,
        data_collator=data_collator,
        beta=0.1,
        # callbacks=[],
    )

    trainer.train()

    model.save_pretrained("./output/dpo/dpo_checkpoint/")
    tokenizer.save_pretrained("./output/dpo/dpo_checkpoint/")