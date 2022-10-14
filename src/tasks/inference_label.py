import json
import torch
import pandas as pd
import ast
from typing import List, Dict, Tuple, Union, Any

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "t5-large"
DATA_PATH = "submissions/sub2_esnli_imp_td.json"
MODEL_PATH = "t5-large-figlang-label"
OUTPUT_PATH = "t5-large-figlang-label"
BATCH_SIZE = 16
MAX_SRC_LENGTH = 256
MAX_TARGET_LENGTH = 16
PREFIX = "mnli "

NUM_BEAMS = 2

SEED = 48
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class Inference:
    def __init__(self, data_path, model_path, output_path, batch_size, prefix, max_src_len, max_target_len):
        self.data_path = data_path
        self.model_path = model_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prefix = prefix
        self.max_src_len = max_src_len
        self.max_target_len = max_target_len
        self.save_results = True

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.config = AutoConfig.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

        self._prepare_data()

    def preprocess_batch(
        self,
        examples,
        premise_column: str,
        hypothesis_column: str,
        explanation_column: str,
    ) -> List[str]:
        premises = examples[premise_column].tolist()
        hypotheses = examples[hypothesis_column].tolist()
        explanations = examples[explanation_column].tolist()

        def generate_input(_premise, _hypothesis, _exp):
            return " ".join(
                ["hypothesis:", _hypothesis.lstrip(), "premise:", _premise.lstrip(), "explanation:", _exp.lstrip()]
            )

        inputs = [
            self.prefix + generate_input(premise, hypothesis, exp)
            for premise, hypothesis, exp in zip(premises, hypotheses, explanations)
        ]
        return inputs

    def _prepare_data(self):
        with open(self.data_path, "r") as f:
            data = f.readlines()
        data = [ast.literal_eval(d) for d in data]
        self.test_data = pd.DataFrame(data)
        inputs = self.preprocess_batch(self.test_data, "premise", "hypothesis", "model_explanation")

        model_inputs = self.tokenizer(
            inputs, max_length=self.max_src_len, padding="max_length", truncation=True
        )
        self.test_data["input_ids"] = model_inputs["input_ids"]
        self.test_data["attention_mask"] = model_inputs["attention_mask"]

    def data_collator(self, features):
        input_ids = [x["input_ids"].tolist() for x in features]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = [x["attention_mask"].tolist() for x in features]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def predict(self):
        predictions = []
        for batch in range(0, len(self.test_data), BATCH_SIZE):
            data = self.test_data[batch: batch + BATCH_SIZE]
            prep_data = self.data_collator([data])
            self.model.eval()
            self.model.to(self.device)
            with torch.no_grad():
                # https://huggingface.co/blog/how-to-generate
                generated_ids = self.model.generate(
                    input_ids=prep_data["input_ids"][0].to(self.device),
                    attention_mask=prep_data["attention_mask"][0].to(self.device),
                    max_length=self.max_target_len,
                    use_cache=True,
                    num_beams=NUM_BEAMS,
                    length_penalty=0.6,
                    early_stopping=True,
                )
            outputs = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            print(outputs)

            if self.save_results:
                for codalab_id, idx, prem, hyp, exp, dec_preds in zip(
                        data["codalab_id"],
                        data["id"],
                        data["premise"],
                        data["hypothesis"],
                        data["model_explanation"],
                        outputs,
                ):
                    predictions.append(
                        {
                            "codalab_id": codalab_id,
                            "id": idx,
                            "premise": prem,
                            "hypothesis": hyp,
                            "predicted_label": "Contradiction"
                            if dec_preds.startswith("contradiction")
                            else "Entailment",
                            "model_explanation": exp,
                        }
                    )
                # print(predictions)
        with open(OUTPUT_PATH + "/outputs_final_task.json", "w") as f:
            f.write(json.dumps(predictions, indent=4))


if __name__ == '__main__':
    inference = Inference(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        batch_size=BATCH_SIZE,
        prefix=PREFIX,
        max_src_len=MAX_SRC_LENGTH,
        max_target_len=MAX_TARGET_LENGTH
    )
    # inference._prepare_data()
    inference.predict()
