from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if __name__ == '__main__':
    model_path = "/home/rachneet/projects/ukp/fig_lang/t5-figlang-model"
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    premise = "I just caught a guy picking up used chewing gum and he put it in his mouth."
    hypothesis = "it was such a pleasant sight to see a guy picking up used chewing gum; and he put it in his mouth"
    prepared_input = f"classify and predict: premise: {premise} hypothesis: {hypothesis}"
    features = tokenizer(prepared_input, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    # print(features)
    outputs = model.generate(**features, max_length=128, num_beams=2)
    dec_preds = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("The prediction is: ", dec_preds)
    print("Contradiction" if dec_preds[:1] == "0" else "Entailment")
    print(dec_preds[1:].replace("explanation:", "").lstrip())
