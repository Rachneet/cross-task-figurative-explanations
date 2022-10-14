from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("figurative-nlp/t5-figurative-paraphrase")
model = AutoModelForSeq2SeqLM.from_pretrained("figurative-nlp/t5-figurative-paraphrase")


input_ids = tokenizer(
    "paraphrase the sentence : The heavy rain guttered the soil.",
    return_tensors="pt"
).input_ids  # Batch size 1
outputs = model.generate(input_ids, num_beams=5, max_length=64)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
