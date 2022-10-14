import json
from typing import List
import torch

from src.preprocessor. dataloader import PreprocessData
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

BATCH_SIZE = 16
NUM_BEAMS = 4
OUTPUT_DIR = "gpt2-figlang-2"


config = GPT2Config.from_pretrained(OUTPUT_DIR, cache_dir=None)
tokenizer = GPT2Tokenizer.from_pretrained(
    OUTPUT_DIR, cache_dir=None,
)
model = GPT2LMHeadModel.from_pretrained(
    OUTPUT_DIR, config=config, cache_dir=None
)
model.cuda()
model.eval()

dataloader = PreprocessData(
    "ColumbiaNLP/FigLang2022SharedTask", save_data=False, save_path=""
)
_, val_set = dataloader.get_processed_data()

for text in val_set:
    # print(text)
    _hypothesis = text["hypothesis"]
    _premise = text["premise"]
    _label = text["label"]
    _exp = text["explanation"]
    prompt = f"<startoftext>hypothesis: {_hypothesis.lstrip()}\npremise: {_premise.lstrip()}" \
                   f"\nlabel:"
    generated = tokenizer(f"{prompt}", return_tensors="pt").cuda()
    sample_outputs = model.generate(
        input_ids=generated["input_ids"],
        attention_mask=generated["attention_mask"],
        do_sample=False, top_k=50, max_length=256, top_p=0.90,
        temperature=0, num_return_sequences=0)
    predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    print(predicted_text)

