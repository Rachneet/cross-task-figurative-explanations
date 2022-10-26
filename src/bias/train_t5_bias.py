"""
Create features for the figurative dataset.
As before, we want it to perform 2 tasks: Binary classification and explanation generation.

For these bias experiments, however, either the premise or the hypothesis will be excluded from the input.
The variable that determines this is BIAS_TYPE ("hyp" or "prem")

Format 1:
Source_text:
    <prefix>: premise: <premise> hypothesis: <hypothesis>
target_text:
    <"0 or 1"> explanation: <explanation>
    Contradiction: 0
    Entailment: 1


Format 2:
Source_text:
    <prefix>:  hypothesis: <hypothesis> premise: <premise>
target_text:
    <label> explanation: <explanation>


Format 3 (hyp-only bias):
Source_text:
    <prefix>:  hypothesis: <hypothesis>
target_text:
    <label> explanation: <explanation>

Format 4 (premise-only bias):
Source_text:
    <prefix>: premise: <premise>
target_text:
    <label> explanation: <explanation>

    
"""

import logging
import json
from typing import List, Tuple
import os

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
)
import transformers.models.t5.modeling_t5
from preprocessor.dataloader import PreprocessData

logger = logging.getLogger(__name__)

# hyperparameters
LEARNING_RATE = 1.0e-4
WARMUP_STEPS = 100
DROPOUT = 0.1
MAX_EPOCHS = 10
FP16 = False
SCHEDULER = "linear"
MODEL_NAME = "t5-large"
BATCH_SIZE = 16
BIAS_TYPE = "hyp"   # "hyp"|"prem"

RESUME_TRAINING = None

OUTPUT_DIR = "t5-" + BIAS_TYPE + "-bias"
CHECKPOINT = None

DO_TRAIN = True
DO_EVAL = True
DO_PREDICT = True

NUM_BEAMS = 4

# set to None when using all data
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

class T5FigurativeModel:
    def __init__(
            self,
            max_src_len,
            max_target_len,
            prefix,
            do_train,
            do_eval,
            do_predict,
            model_name,
            cache_dir=None,
            max_train_samples=None,
            max_val_samples=None,
            save_results=False
    ):
        self.max_src_len = max_src_len
        self.max_target_len = max_target_len
        self.prefix = prefix
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_predict = do_predict
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.padding = "max_length"
        self.save_results = save_results
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load resources
        self._prepare_data()

    def _prepare_data(self):

        config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            config=config,
            cache_dir=self.cache_dir,
        )
        dataloader = PreprocessData(
            "ColumbiaNLP/FigLang2022SharedTask",
            save_data=False,
            save_path=""
        )
        self.train_set, self.val_set = dataloader.get_processed_data()

    def preprocess_train_batch(
        self,
        examples,
        premise_column: str,
        hypothesis_column: str,
        label_column: str,
        explanation_column: str,
    ) -> Tuple[List[str], List[str]]:
        premises = examples[premise_column]
        hypotheses = examples[hypothesis_column]
        labels = examples[label_column]
        explanations = examples[explanation_column]

        def generate_input(_premise, _hypothesis):
            if BIAS_TYPE == "hyp":
                return " ".join(["hypothesis:", _hypothesis.lstrip()])
            elif BIAS_TYPE == "prem":
                return " ".join(["premise:", _premise.lstrip()])

        def generate_target_input(_label, _exp):
            return " ".join([_label.lower(), "explanation:", _exp.lstrip()])

        inputs = [self.prefix + generate_input(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]
        # print(inputs)
        targets = [generate_target_input(label, exp) for label, exp in zip(labels, explanations)]
        # print(targets)
        return inputs, targets

    def preprocess_val_batch(
        self,
        examples,
        premise_column: str,
        hypothesis_column: str,
        label_column: str,
        explanation_column: str,
    ) -> Tuple[List[str], List[str]]:
        premises = examples[premise_column]
        hypotheses = examples[hypothesis_column]
        labels = examples[label_column]
        explanations = examples[explanation_column]

        def generate_input(_premise, _hypothesis):
            return " ".join(["hypothesis:", _hypothesis.lstrip(), "premise:", _premise.lstrip()])

        def generate_target_input(_label, _exp):
            return " ".join([_label.lower(), "explanation:", _exp.lstrip()])

        inputs = [self.prefix + generate_input(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]
        # print(inputs)
        targets = [generate_target_input(label, exp) for label, exp in zip(labels, explanations)]
        # print(targets)
        return inputs, targets


    def _prepare_train_features(self, examples):
        if self.do_train:
            column_names = self.train_set.column_names
        elif self.do_eval or self.do_predict:
            column_names = self.val_set.column_names

        # ['id', 'premise', 'hypothesis', 'label', 'explanation', 'split', 'type', 'idiom']
        prem_column_name = "premise" if "premise" in column_names else column_names[1]
        hyp_column_name = "hypothesis" if "hypothesis" in column_names else column_names[2]
        label_column_name = "label" if "label" in column_names else column_names[3]
        exp_column_name = "explanation" if "explanation" in column_names else column_names[4]

        inputs, targets = self.preprocess_train_batch(
                examples,
                prem_column_name,
                hyp_column_name,
                label_column_name,
                exp_column_name
        )

        model_inputs = self.tokenizer(inputs, max_length=self.max_src_len, padding=self.padding, truncation=True)
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_len, padding=self.padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def _prepare_val_features(self, examples):
        if self.do_train:
            column_names = self.train_set.column_names
        elif self.do_eval or self.do_predict:
            column_names = self.val_set.column_names

        # ['id', 'premise', 'hypothesis', 'label', 'explanation', 'split', 'type', 'idiom']
        prem_column_name = "premise" if "premise" in column_names else column_names[1]
        hyp_column_name = "hypothesis" if "hypothesis" in column_names else column_names[2]
        label_column_name = "label" if "label" in column_names else column_names[3]
        exp_column_name = "explanation" if "explanation" in column_names else column_names[4]
   
        inputs, targets = self.preprocess_val_batch(
                examples,
                prem_column_name,
                hyp_column_name,
                label_column_name,
                exp_column_name
        )

        model_inputs = self.tokenizer(inputs, max_length=self.max_src_len, padding=self.padding, truncation=True)
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_len, padding=self.padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def data_collator(self, features):
        input_ids = [x["input_ids"] for x in features]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = [x["attention_mask"] for x in features]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = [x["labels"] for x in features]
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def train(self):
        if self.do_train:
            logger.info("*** Train ***")
            if self.max_train_samples is not None:
                # We will select sample from whole data if argument is specified
                max_train_samples = min(len(self.train_set), self.max_train_samples)
                self.train_set = self.train_set.select(range(max_train_samples))
            train_dataset = self.train_set.map(
                self._prepare_train_features,    
                batched=True,
                desc="Running tokenizer on train dataset",
            )
        if self.do_eval or self.do_predict:
            if self.max_val_samples is not None:
                # We will select sample from whole data if argument is specified
                max_val_samples = min(len(self.val_set), self.max_val_samples)
                self.val_set = self.val_set.select(range(max_val_samples))
            eval_dataset = self.val_set.map(
                self._prepare_val_features,             
                batched=True,
                desc="Running tokenizer on validation dataset",
            )

        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            # eval_steps=4000,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            lr_scheduler_type=SCHEDULER,
            num_train_epochs=MAX_EPOCHS,
            logging_steps=10,
            save_steps=250,
            disable_tqdm=False,
            remove_unused_columns=False,
            fp16=FP16,
            label_names=[
                "labels",
            ],  # it's important to log eval_loss
        )
        print("Batch Size", args.train_batch_size)
        print("Parallel Mode", args.parallel_mode)

        trainer = Trainer(
            model=self.model,
            args=args,
            data_collator=self.data_collator,
            train_dataset=train_dataset if self.do_train else None,
            eval_dataset=eval_dataset if self.do_eval else None,
        )
        try:
            if self.do_train:
                checkpoint = None
                if RESUME_TRAINING is not None:
                    checkpoint = RESUME_TRAINING
                trainer.train(resume_from_checkpoint=checkpoint)
                trainer.save_model()
        except KeyboardInterrupt:
            trainer.save_model("interrupted-fig-lang")

        if self.do_predict:
            logger.info("*** Predict ***")
            # print(eval_dataset)
            # print(eval_dataset[0])
            if CHECKPOINT is not None:
                model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
            else:
                model = self.model
            predictions = []
            for batch in range(0, len(eval_dataset), BATCH_SIZE):
                data = eval_dataset[batch: batch+BATCH_SIZE]
                # print(data)
                prep_data = self.data_collator([data])
                model.eval()
                model.to(self.device)
                with torch.no_grad():
                    # https://huggingface.co/blog/how-to-generate
                    generated_ids = model.generate(
                        input_ids=prep_data['input_ids'][0].to(self.device),
                        attention_mask=prep_data['attention_mask'][0].to(self.device),
                        max_length=self.max_target_len,
                        use_cache=True,
                        num_beams=NUM_BEAMS,
                        length_penalty=0.6,
                        early_stopping=True
                    )
                outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                if self.save_results:
                    for prem, hyp, label, exp, dec_preds in zip(
                            data["premise"],
                            data["hypothesis"],
                            data["label"],
                            data["explanation"],
                            outputs
                    ):
                        predictions.append({
                            "premise": prem,
                            "hypothesis": hyp,
                            "label": label,
                            "explanation": exp,
                            "predicted_label": "Contradiction" if dec_preds.startswith("contradiction") else "Entailment",
                            "model_explanation": " ".join(dec_preds.split(" ")[2:]).lstrip()
                        }
                    )
                    # print(predictions)
            with open(OUTPUT_DIR+"/outputs_esnli_bias.json", "w") as f:
                f.write(json.dumps(predictions, indent=4))


if __name__ == '__main__':
    trainer = T5FigurativeModel(
        max_src_len=128,
        max_target_len=128,
        prefix="figurative ",
        do_train=DO_TRAIN,
        do_eval=DO_EVAL,
        do_predict=DO_PREDICT,
        model_name=MODEL_NAME,
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_val_samples=MAX_VAL_SAMPLES,
        save_results=True
    )
    trainer.train()
