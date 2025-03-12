import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

" Load model, tokenizer "
model = AutoModelForCausalLM.from_pretrained(
    'output/sft/checkpoint-1174/' # , device_map='cpu'
)
tokenizer= AutoTokenizer.from_pretrained(
    '/root/autodl-tmp/NL2SQL/models/Qwen2.5-Coder-1.5B'
)


" Load test data "
