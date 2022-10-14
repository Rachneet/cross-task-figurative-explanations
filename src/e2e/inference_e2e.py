from copy import deepcopy
from typing import Tuple, List
import torch
import ast
import pandas as pd
import json

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
)

USE_PREDICTED_LABELS = True
NUM_BEAMS = 4
BATCH_SIZE = 16

DATA_PATH = "data/answer.json"
MODEL_PATH = "src/t5-large-e2e-impli-td-0.1-0.9/t5-large-e2e-impli-td-0.1-0.9"
OUTPUT_PATH = "src/t5-large-e2e-impli-td-0.1-0.9"

MAX_SRC_LEN = 128
MAX_PRED_LEN = 6
MAX_TARGET_LENGTH = 128

PREFIX = "figurative "


class Inference:
    def __init__(self, data_path, model_path, output_path, batch_size, prefix, max_src_len,
                 max_pred_len, max_target_len):
        self.data_path = data_path
        self.model_path = model_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prefix = prefix
        self.max_src_len = max_src_len
        self.max_pred_len = max_pred_len
        self.max_target_len = max_target_len
        self.save_results = True

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.config = AutoConfig.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

        self._prepare_data()

    def _prepare_data(self):
        with open(self.data_path, "r") as f:
            data = f.readlines()
        data = [ast.literal_eval(d) for d in data]
        self.test_data = pd.DataFrame(data)

    def preprocess_batch(
            self,
            examples,
            premise_column: str,
            hypothesis_column: str,
            label_column: str,
            phase_one: bool,
            phase_2_generate: bool,
    ) -> List[str]:

        premises = examples[premise_column].tolist()
        hypotheses = examples[hypothesis_column].tolist()

        if phase_one:
            def generate_input(_premise, _hypothesis):
                return " ".join(["premise:", _premise.lstrip(), "hypothesis:", _hypothesis.lstrip()])

            inputs = [self.prefix + generate_input(premise, hypothesis) for premise, hypothesis in
                      zip(premises, hypotheses)]
            return inputs

        if phase_2_generate:
            labels = examples[label_column].tolist()

            def generate_input(_premise, _hypothesis, label):
                return " ".join(["premise:", _premise.lstrip(), "hypothesis:", _hypothesis.lstrip(), label])

            inputs = [self.prefix + generate_input(premise, hypothesis, label) for premise, hypothesis, label in
                  zip(premises, hypotheses, labels)]
            return inputs

    def _get_predictions(self, args):
        combined_inputs = args['combined_inputs']
        prem_column_name = args['prem_column_name']
        hyp_column_name = args['hyp_column_name']
        label_column_name = args['label_column_name']

        phase_1 = args['phase_1']

        phase_2_generate = False
        if 'phase_2_generate' in args.keys() and args['phase_2_generate']:
            phase_2_generate = True

        inputs = self.preprocess_batch(
            combined_inputs,
            prem_column_name,
            hyp_column_name,
            label_column_name,
            phase_1,
            phase_2_generate
        )

        model_inputs = self.tokenizer(inputs, max_length=self.max_src_len,
                                      padding="max_length", truncation=True, return_tensors="pt")
        model_inputs = model_inputs.to(self.device)
        return model_inputs

    def _get_two_phase_output(self, **args):
        ## model will be set to no grad, eval cause we pass it labels.

        args['phase_1'] = True
        model_inputs = self._get_predictions(args)  ## For phase 1

        if 'phase_2_generate' in args.keys() and args['phase_2_generate']:
            phase_2_generate = True
        else:
            phase_2_generate = False

        predictions = None
        generated_ids = None
        if USE_PREDICTED_LABELS:

            generated_ids = self.model.generate(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                max_length=self.max_pred_len,
                use_cache=False,
                num_beams=NUM_BEAMS,
                early_stopping=True
            )
            predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            combined_inputs = deepcopy(args['combined_inputs'])
            if phase_2_generate:
                assert len(combined_inputs[list(combined_inputs.keys())[0]]) == len(predictions)
                combined_inputs[args['label_column_name']] = predictions
            else:
                assert len(combined_inputs) == len(predictions)

                for i in range(len(predictions)):
                    combined_inputs[i][args['label_column_name']] = predictions[i]

            args['combined_inputs'] = combined_inputs

        args['phase_1'] = False

        ## phase_2_generate is true only during prediction
        ##     --> where we use beam search instead of using the output logits
        if phase_2_generate:
            model_inputs = self._get_predictions(args)  ## Overloaded with "phase_2_generate"
            generated_ids = self.model.generate(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                max_length=self.max_target_len,
                use_cache=True,
                num_beams=NUM_BEAMS,
                early_stopping=True
            )

            return generated_ids, predictions  ## Phase 1 preds

        outputs, model_inputs = self._get_predictions(args)  ## For phase 2
        labels = model_inputs["labels"]

        return outputs, labels

    def predict(self):
        predictions = []
        for batch in range(0, len(self.test_data), BATCH_SIZE):
            data = self.test_data[batch: batch + BATCH_SIZE]
            # prep_data = self.data_collator([data])
            self.model.eval()
            self.model.to(self.device)
            with torch.no_grad():
                generated_ids, labels = self._get_two_phase_output(
                    combined_inputs=data,
                    prem_column_name="premise",
                    hyp_column_name="hypothesis",
                    label_column_name="label",
                    phase_2_generate=True
                )
            outputs = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            if self.save_results:
                for codalab_id, idx, prem, hyp, pred_label, dec_preds in zip(
                        data["codalab_id"],
                        data["id"],
                        data["premise"],
                        data["hypothesis"],
                        labels,
                        outputs,
                ):
                    predictions.append(
                        {
                            "codalab_id": codalab_id,
                            "id": idx,
                            "premise": prem,
                            "hypothesis": hyp,
                            "predicted_label": "Contradiction"
                            if pred_label.startswith("contradiction")
                            else "Entailment",
                            "model_explanation":  dec_preds.replace("explanation:", "").lstrip(),
                        }
                    )
                # print(predictions)
        with open(OUTPUT_PATH+"/outputs_final_task.json", "w") as f:
            f.write(json.dumps(predictions, indent=4))


if __name__ == '__main__':
    inference = Inference(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        batch_size=BATCH_SIZE,
        prefix=PREFIX,
        max_src_len=MAX_SRC_LEN,
        max_pred_len=MAX_PRED_LEN,
        max_target_len=MAX_TARGET_LENGTH
    )
    # inference._prepare_data()
    inference.predict()
