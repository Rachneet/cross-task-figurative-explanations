"""
Create features for the figurative dataset
We want it to perform 2 tasks: Binary classification and generative (for explanations)

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
"""

import logging
import json
from typing import List, Tuple

import torch
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)
import wandb
from src.preprocessor.dataloader import PreprocessData

logger = logging.getLogger(__name__)

# hyperparameters
LEARNING_RATE = 1.0e-4
WARMUP_STEPS = 100
DROPOUT = 0.1
MAX_EPOCHS = 10
FP16 = False
SCHEDULER = "linear"
MODEL_NAME = "gpt2"
BATCH_SIZE = 16

RESUME_TRAINING = None

OUTPUT_DIR = "gpt2-figlang-2"
CHECKPOINT = None

DO_TRAIN = True
DO_EVAL = True

NUM_BEAMS = 4

# set to None when using all data
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

WANDB_RUN_NAME = OUTPUT_DIR

wandb.init(
    config={
        "lr": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "dropout": 0.1,
        "epochs": MAX_EPOCHS,
        "scheduler": SCHEDULER,
        "model": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "output_dir": OUTPUT_DIR,
        "num_beams": NUM_BEAMS,
    }
)
wandb.run.name = WANDB_RUN_NAME

SEED = 48
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class GPTFigurativeModel:
    def __init__(
        self,
        max_src_len,
        max_target_len,
        prefix,
        do_train,
        do_eval,
        model_name,
        cache_dir=None,
        max_train_samples=None,
        max_val_samples=None,
        save_results=False,
    ):
        self.max_src_len = max_src_len
        self.max_target_len = max_target_len
        self.prefix = prefix
        self.do_train = do_train
        self.do_eval = do_eval
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

        config = GPT2Config.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.model_name, pad_token="<pad>",
            bos_token="<startoftext>", eos_token="<endoftext>", cache_dir=self.cache_dir,
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name, config=config, cache_dir=self.cache_dir
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        dataloader = PreprocessData(
            "ColumbiaNLP/FigLang2022SharedTask", save_data=False, save_path=""
        )
        self.train_set, self.val_set = dataloader.get_processed_data()

    def preprocess_batch(
        self,
        examples,
        premise_column: str,
        hypothesis_column: str,
        label_column: str,
        explanation_column: str,
    ) -> List[str]:
        premises = examples[premise_column]
        hypotheses = examples[hypothesis_column]
        labels = examples[label_column]
        explanations = examples[explanation_column]

        def generate_input(_premise, _hypothesis, _label, _exp):
            return f"hypothesis: {_hypothesis.lstrip()}\npremise: {_premise.lstrip()}" \
                   f"\nlabel: {_label.lstrip()}\nexplanation: {_exp.lstrip()}"

        inputs = [
            "<startoftext>" + generate_input(premise, hypothesis, label, exp) + "<endoftext>"
            for premise, hypothesis, label, exp in zip(premises, hypotheses, labels, explanations)
        ]
        return inputs

    def _prepare_features(self, examples):
        if self.do_train:
            column_names = self.train_set.column_names
        elif self.do_eval:
            column_names = self.val_set.column_names

        # ['id', 'premise', 'hypothesis', 'label', 'explanation', 'split', 'type', 'idiom']
        prem_column_name = "premise" if "premise" in column_names else column_names[1]
        hyp_column_name = (
            "hypothesis" if "hypothesis" in column_names else column_names[2]
        )
        label_column_name = "label" if "label" in column_names else column_names[3]
        exp_column_name = (
            "explanation" if "explanation" in column_names else column_names[4]
        )

        inputs = self.preprocess_batch(
            examples,
            prem_column_name,
            hyp_column_name,
            label_column_name,
            exp_column_name,
        )

        model_inputs = self.tokenizer(
            inputs, max_length=self.max_src_len, padding=self.padding, truncation=True
        )
        return model_inputs

    def data_collator(self, features):
        input_ids = [x["input_ids"] for x in features]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = [x["attention_mask"] for x in features]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        # we pass the input data as the label instead of just the NLI labels.
        # This is because we are training a language model, hence we want the
        # model to learn the pattern of the prompt and not just NLI class.
        labels = [x["input_ids"] for x in features]
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def train(self):
        if self.do_train:
            logger.info("*** Train ***")
            if self.max_train_samples is not None:
                # We will select sample from whole data if argument is specified
                max_train_samples = min(len(self.train_set), self.max_train_samples)
                self.train_set = self.train_set.select(range(max_train_samples))
            train_dataset = self.train_set.map(
                self._prepare_features,
                batched=True,
                desc="Running tokenizer on train dataset",
            )
        if self.do_eval:
            if self.max_val_samples is not None:
                # We will select sample from whole data if argument is specified
                max_val_samples = min(len(self.val_set), self.max_val_samples)
                self.val_set = self.val_set.select(range(max_val_samples))
            eval_dataset = self.val_set.map(
                self._prepare_features,
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
            save_total_limit=2,
            save_strategy="epoch",
            load_best_model_at_end=True,
            run_name=WANDB_RUN_NAME,
            disable_tqdm=False,
            report_to=["wandb"],
            remove_unused_columns=False,
            fp16=FP16,
            seed=SEED,
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
                self.tokenizer.save_pretrained(OUTPUT_DIR)
        except KeyboardInterrupt:
            trainer.save_model("interrupted-fig-lang")

        wandb.finish()


if __name__ == "__main__":
    trainer = GPTFigurativeModel(
        max_src_len=256,
        max_target_len=256,
        prefix="",
        do_train=DO_TRAIN,
        do_eval=DO_EVAL,
        model_name=MODEL_NAME,
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_val_samples=MAX_VAL_SAMPLES,
        save_results=True,
    )
    trainer.train()
