"""
Hyperparameter optimization task using wandb sweeps
"""

import logging
import json
import numpy as np
from typing import List, Tuple, Callable, Iterable

from datasets import load_metric
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

import wandb
from src.tasks.figlang_dataloader import PreprocessData

logger = logging.getLogger(__name__)

# hyperparameters
LEARNING_RATE = 1.0e-4
WARMUP_STEPS = 100
DROPOUT = 0.1
MAX_EPOCHS = 1
FP16 = False
SCHEDULER = "linear"
MODEL_NAME = "t5-small"
BATCH_SIZE = 16

RESUME_TRAINING = None

OUTPUT_DIR = "t5-small-figlang"
CHECKPOINT = None

DO_TRAIN = True
DO_EVAL = True
DO_PREDICT = False

NUM_BEAMS = 4

# set to None when using all data
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

# WANDB_RUN_NAME = OUTPUT_DIR

SEED = 48
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# method
sweep_config = {
    'method': 'random'
}

# hyperparameters
parameters_dict = {
    'epochs': {
        'value': 1
        },
    'batch_size': {
        'values': [8, 16, 32, 64]
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-3
    },
    'weight_decay': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
}

sweep_config['parameters'] = parameters_dict

metric = {
    'name': 'eval/loss',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric
sweep_id = wandb.sweep(sweep_config, project='t5-figlang-sweeps')


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
        save_results=False,
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

        config = AutoConfig.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, use_fast=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, config=config, cache_dir=self.cache_dir
        )
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
    ) -> Tuple[List[str], List[str]]:
        premises = examples[premise_column]
        hypotheses = examples[hypothesis_column]
        labels = examples[label_column]
        explanations = examples[explanation_column]

        def generate_input(_premise, _hypothesis):
            return " ".join(
                ["hypothesis:", _hypothesis.lstrip(), "premise:", _premise.lstrip()]
            )

        def generate_target_input(_label, _exp):
            return " ".join([_label.lower(), "explanation:", _exp.lstrip()])

        inputs = [
            self.prefix + generate_input(premise, hypothesis)
            for premise, hypothesis in zip(premises, hypotheses)
        ]
        # print(inputs)
        targets = [
            generate_target_input(label, exp)
            for label, exp in zip(labels, explanations)
        ]
        # print(targets)
        return inputs, targets

    def _prepare_features(self, examples):
        if self.do_train:
            column_names = self.train_set.column_names
        elif self.do_eval or self.do_predict:
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

        inputs, targets = self.preprocess_batch(
            examples,
            prem_column_name,
            hyp_column_name,
            label_column_name,
            exp_column_name,
        )

        model_inputs = self.tokenizer(
            inputs, max_length=self.max_src_len, padding=self.padding, truncation=True
        )
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.max_target_len,
                padding=self.padding,
                truncation=True,
            )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
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
            "labels": labels,
        }

    def compute_metrics(self, eval_preds):

        label_2_id = {"contradiction": 0, "entailment": 1}
        pred_ids = eval_preds.predictions
        label_ids = eval_preds.label_ids

        pred_str = self.tokenizer.batch_decode(
            torch.tensor(pred_ids, device=self.model.device), skip_special_tokens=True)
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(
            torch.tensor(label_ids, device=self.model.device), skip_special_tokens=True)

        labels = [label_2_id.get(label.split()[0]) for label in label_str]
        preds = [label_2_id.get(pred.split()[0], 2) for pred in pred_str]

        metrics = dict()

        accuracy_metric = load_metric('accuracy')
        precision_metric = load_metric('precision')
        recall_metric = load_metric('recall', zero_division=0)
        f1_metric = load_metric('f1')

        metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
        metrics.update(precision_metric.compute(predictions=preds, references=labels, average='weighted'))
        metrics.update(recall_metric.compute(predictions=preds, references=labels, average='weighted'))
        metrics.update(f1_metric.compute(predictions=preds, references=labels, average='weighted'))

        # log metrics to wandb
        wandb.log({
            "accuracy": metrics["accuracy"],
        })

        return metrics

    def train(self, config=None):
        with wandb.init(
                config=config,
                # settings=wandb.Settings(console='off')
        ):
            config = wandb.config
            # wandb.run.name = WANDB_RUN_NAME

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
            if self.do_eval or self.do_predict:
                if self.max_val_samples is not None:
                    # We will select sample from whole data if argument is specified
                    max_val_samples = min(len(self.val_set), self.max_val_samples)
                    self.val_set = self.val_set.select(range(max_val_samples))
                eval_dataset = self.val_set.map(
                    self._prepare_features,
                    batched=True,
                    desc="Running tokenizer on validation dataset",
                )

            args = Seq2SeqTrainingArguments(
                output_dir=OUTPUT_DIR,
                overwrite_output_dir=False,
                do_train=self.do_train,
                do_eval=self.do_eval,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=BATCH_SIZE,
                learning_rate=config.learning_rate,
                num_train_epochs=config.epochs,
                weight_decay=config.weight_decay,
                lr_scheduler_type=SCHEDULER,
                save_strategy='epoch',
                evaluation_strategy='epoch',
                logging_strategy='epoch',
                predict_with_generate=True,
                load_best_model_at_end=True,
                save_total_limit=2,
                # run_name=WANDB_RUN_NAME,
                disable_tqdm=False,
                report_to=["wandb"],
                remove_unused_columns=False,
                fp16=FP16,
                seed=SEED,
                label_names=["labels"],  # it's important to log eval_loss
            )
            # print("Batch Size", args.train_batch_size)
            # print("Parallel Mode", args.parallel_mode)

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=args,
                data_collator=self.data_collator,
                train_dataset=train_dataset if self.do_train else None,
                eval_dataset=eval_dataset if self.do_eval else None,
                compute_metrics=self.compute_metrics,
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
                if CHECKPOINT is not None:
                    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
                else:
                    model = self.model
                predictions = []
                for batch in range(0, len(eval_dataset), BATCH_SIZE):
                    data = eval_dataset[batch: batch + BATCH_SIZE]
                    prep_data = self.data_collator([data])
                    model.eval()
                    model.to(self.device)
                    with torch.no_grad():
                        # https://huggingface.co/blog/how-to-generate
                        generated_ids = model.generate(
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
                                    "predicted_label": "Contradiction"
                                    if dec_preds.startswith("contradiction")
                                    else "Entailment",
                                    "model_explanation": " ".join(
                                        dec_preds.split(" ")[2:]
                                    ).lstrip(),
                                }
                            )
                        # print(predictions)
                with open(OUTPUT_DIR + "/outputs.json", "w") as f:
                    f.write(json.dumps(predictions, indent=4))


if __name__ == "__main__":
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
        save_results=True,
    )
    wandb.agent(sweep_id, function=trainer.train, count=20)
    wandb.finish()
    # trainer.train()
