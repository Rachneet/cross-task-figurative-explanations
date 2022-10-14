import json
from typing import List
import torch

from src.preprocessor. dataloader import PreprocessData
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

BATCH_SIZE = 16
NUM_BEAMS = 4
OUTPUT_DIR = "gpt2-figlang-2"

class Inference:

    def __init__(self, model_path, max_src_len, max_tgt_len, save_results):
        self.model_path = model_path
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.save_results = save_results
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _prepare_data(self):
        config = GPT2Config.from_pretrained(self.model_path, cache_dir=None)
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.model_path,
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_path, config=config, cache_dir=None
        )
        # self.model.resize_token_embeddings(len(self.tokenizer))
        dataloader = PreprocessData(
            "ColumbiaNLP/FigLang2022SharedTask", save_data=False, save_path=""
        )
        _, self.val_set = dataloader.get_processed_data()

    def preprocess_batch(
            self,
            examples,
            premise_column: str,
            hypothesis_column: str,
    ) -> List[str]:
        premises = examples[premise_column]
        hypotheses = examples[hypothesis_column]

        def generate_input(_premise, _hypothesis):
            return f"hypothesis: {_hypothesis.lstrip()}\npremise: {_premise.lstrip()}\nlabel:"

        inputs = [
            "<startoftext>" + generate_input(premise, hypothesis)
            for premise, hypothesis in zip(premises, hypotheses)
        ]
        return inputs

    def _prepare_features(self, examples):

        column_names = self.val_set.column_names
        # ['id', 'premise', 'hypothesis', 'label', 'explanation', 'split', 'type', 'idiom']
        prem_column_name = "premise" if "premise" in column_names else column_names[1]
        hyp_column_name = (
            "hypothesis" if "hypothesis" in column_names else column_names[2]
        )
        inputs = self.preprocess_batch(
            examples,
            prem_column_name,
            hyp_column_name,
        )
        model_inputs = self.tokenizer(
            inputs, max_length=self.max_src_len, padding="max_length", truncation=True
        )
        return model_inputs

    def data_collator(self, features):
        input_ids = [x["input_ids"] for x in features]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = [x["attention_mask"] for x in features]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def predict(self):
        self._prepare_data()
        eval_dataset = self.val_set.map(
            self._prepare_features,
            batched=True,
            desc="Running tokenizer on validation dataset",
        )
        predictions = []
        for batch in range(0, len(eval_dataset), BATCH_SIZE):
            data = eval_dataset[batch: batch + BATCH_SIZE]
            print(data)
            prep_data = self.data_collator([data])
            self.model.eval()
            self.model.to(self.device)
            with torch.no_grad():
                # https://huggingface.co/blog/how-to-generate
                generated_ids = self.model.generate(
                    input_ids=prep_data["input_ids"][0].to(self.device),
                    attention_mask=prep_data["attention_mask"][0].to(self.device),
                    max_length=self.max_tgt_len,
                    top_k=50,
                    top_p=0.90,
                    num_return_sequences=0,
                )
            outputs = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            if self.save_results:
                for prem, hyp, label, exp, dec_preds in zip(
                        data["premise"],
                        data["hypothesis"],
                        data["label"],
                        data["explanation"],
                        outputs,
                ):
                    predictions.append(
                        {
                            "premise": prem,
                            "hypothesis": hyp,
                            "label": label,
                            "explanation": exp,
                            "model_prediction": dec_preds.lstrip(),
                        }
                    )
                # print(predictions)
        with open(OUTPUT_DIR + "/outputs.json", "w") as f:
            f.write(json.dumps(predictions, indent=4))


if __name__ == '__main__':
    inf = Inference(
        model_path=OUTPUT_DIR,
        max_src_len=256,
        max_tgt_len=512,
        save_results=True
    )
    inf.predict()
