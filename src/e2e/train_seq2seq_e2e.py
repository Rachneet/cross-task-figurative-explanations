"""
End2End model for the generation of NLI and explanations of figurative language.
"""

import os
import sys
import logging
import json
from typing import List, Tuple

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.generation_logits_process import LogitsProcessorList, MinLengthLogitsProcessor

from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)

################################################################################
##                        Hyperparams                                         ##
################################################################################

WANDB = True

USE_PREDICTED_LABELS = True  ## If set to false, will use gold labels during training. NOTE: Must change to true during eval.
NUM_BEAMS = 4

# set to None when using all data
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
DROPOUT = 0.1
FP16 = False
SCHEDULER = "linear"
BATCH_SIZE = 16

################################################################################
##             THESE ARE OVERWRITTEN BY COMMAND LINE ARGUMENTS                ##
################################################################################

DO_TRAIN = True
DO_EVAL = True
DO_PREDICT = True

MAX_EPOCHS = 10

DATASET_TO_USE = "IMPLI"  ## IMPLI|eSNLI|eFigSNLI

# (alpha * explanation_loss) + (1-alpha)*label_loss, higher alpha is more importance to explanations
# Will be ignored if DATASET_TO_USE == "IMPLI"
ALPHA = 0.9

RUN_NAME = "t5-large-e2e-esnli-impli-td-0.1-0.9"
OUTPUT_DIR = "/content/drive/MyDrive/TMP/" + RUN_NAME
BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/figlang"
IMPLI_DATA_LOCATION = BASE_PATH+"/data/impli_full/"

MODEL_NAME = "t5-base"

RESUME_TRAINING = None
CHECKPOINT = None

DATA_SOURCE = None

WANDB_RUN_NAME = RUN_NAME
eFigSNLI_DATA_SOURCE = "ColumbiaNLP/FigLang2022SharedTask"

if WANDB:
    import wandb

    wandb.init(config={"lr": LEARNING_RATE, "warmup_steps": WARMUP_STEPS,
                       "dropout": 0.1, "epochs": MAX_EPOCHS, "scheduler": SCHEDULER,
                       "model": MODEL_NAME, "batch_size": BATCH_SIZE, "output_dir": OUTPUT_DIR,
                       "num_beams": NUM_BEAMS})
    wandb.run.name = WANDB_RUN_NAME


def preprocess_batch(
        examples,
        prefix: str,
        premise_column: str,
        hypothesis_column: str,
        label_column: str,
        explanation_column: str,
        phase_one: bool,
        phase_2_generate: bool,
) -> Tuple[List[str], List[str]]:
    explanations = None
    if phase_2_generate:
        labels = examples[label_column]
        premises = examples[premise_column]
        hypotheses = examples[hypothesis_column]
        if not explanation_column is None:
            explanations = examples[explanation_column]
    else:
        labels = [e[label_column] for e in examples]
        premises = [e[premise_column] for e in examples]
        hypotheses = [e[hypothesis_column] for e in examples]
        if not explanation_column is None:
            explanations = [e[explanation_column] for e in examples]

    if DATASET_TO_USE == 'eSNLI':
        labels = ["entailment" if l == 0 else 'contradiction' for l in labels]

    if phase_one:
        def generate_input(_premise, _hypothesis):
            return " ".join(["premise:", _premise.lstrip(), "hypothesis:", _hypothesis.lstrip()])

        def generate_target_input(_label, ):
            return _label

        inputs = [prefix + generate_input(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]
        targets = [generate_target_input(label) for label in labels]
        return inputs, targets

    if explanations is None:
        raise Exception("Attempting to predict explanations on task without explanations!")

    ## Phase 2, note that the labels must be predictions
    def generate_input(_premise, _hypothesis, label):
        return " ".join(["premise:", _premise.lstrip(), "hypothesis:", _hypothesis.lstrip(), label])

    def generate_target_input(_exp):
        return " ".join(["explanation:", _exp.lstrip()])

    inputs = [prefix + generate_input(premise, hypothesis, label) for premise, hypothesis, label in
              zip(premises, hypotheses, labels)]
    targets = [generate_target_input(exp) for exp in explanations]
    return inputs, targets


def _get_predictions_and_loss(args):
    combined_inputs = args['combined_inputs']
    prefix = args['prefix']
    prem_column_name = args['prem_column_name']
    hyp_column_name = args['hyp_column_name']
    label_column_name = args['label_column_name']
    exp_column_name = args['exp_column_name']
    tokenizer = args['tokenizer']
    max_pred_len = args['max_pred_len']
    max_src_len = args['max_src_len']
    max_target_len = args['max_target_len']
    padding = args['padding']
    device = args['device']
    phase_1 = args['phase_1']
    model = args['model']

    phase_2_generate = False
    if 'phase_2_generate' in args.keys() and args['phase_2_generate']:
        phase_2_generate = True

    inputs, targets = preprocess_batch(
        combined_inputs,
        prefix,
        prem_column_name,
        hyp_column_name,
        label_column_name,
        exp_column_name,
        phase_1,
        phase_2_generate
    )

    model_inputs = tokenizer(inputs, max_length=max_src_len, padding=padding, truncation=True, return_tensors="pt")
    model_inputs = model_inputs.to(device)

    if phase_2_generate and not phase_1:
        return model_inputs

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        max_len = max_pred_len if phase_1 else max_target_len
        labels = tokenizer(targets, max_length=max_len, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    labels = labels["input_ids"]

    ## Inputs: ['input_ids', 'attention_mask', 'labels']
    ##   Each a tensor
    # forward pass

    labels = [torch.tensor(l) for l in labels]
    model_inputs["labels"] = torch.stack(labels).to(device)
    outputs = model(**model_inputs)

    return outputs, model_inputs


def _get_two_phase_output(**args):
    ## model will be set to no grad, eval cause we pass it labels.

    args['phase_1'] = True
    outputs, model_inputs = _get_predictions_and_loss(args)  ## For phase 1
    loss_labels = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    if 'phase_2_generate' in args.keys() and args['phase_2_generate']:
        phase_2_generate = True
    else:
        phase_2_generate = False

    predictions = None
    generated_ids = None
    if USE_PREDICTED_LABELS:

        ## Will not use this as we need greedy!
        ## This is from greedy search because we have only one token:
        ##    https://github.com/huggingface/transformers/blob/a9eee2ffecc874df7dd635b2c6abb246fdb318cc/src/transformers/generation_utils.py#L1631

        # next_token_logits = outputs.logits[:, -1, :]

        # logits_processor = LogitsProcessorList(
        #     [
        #         MinLengthLogitsProcessor(10, eos_token_id=args[ 'model' ].config.eos_token_id),
        #     ]
        # )

        # input_ids = model_inputs.input_ids
        # next_tokens_scores = logits_processor(input_ids, next_token_logits)
        # next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        # predictions = args[ 'tokenizer' ].batch_decode(next_tokens, skip_special_tokens=True)
        ## End of

        generated_ids = args['model'].generate(
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            max_length=args['max_pred_len'],
            use_cache=False,
            num_beams=NUM_BEAMS,
            early_stopping=True
        )
        predictions = args['tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)

        ## Greedy:
        # generated_ids = args[ 'model' ].generate(
        #     input_ids=model_inputs['input_ids'],
        #     attention_mask=model_inputs['attention_mask'],
        #     max_length=args[ 'max_pred_len' ],
        #     use_cache=True,
        #     early_stopping=True
        # )
        # outputs = args[ 'tokenizer' ].batch_decode(generated_ids, skip_special_tokens=True)
        # print( "Greedy:" )
        # print( outputs )

        combined_inputs = deepcopy(args['combined_inputs'])
        if phase_2_generate:
            assert len(combined_inputs[list(combined_inputs.keys())[0]]) == len(predictions)
            combined_inputs[args['label_column_name']] = predictions
        else:
            assert len(combined_inputs) == len(predictions)

            for i in range(len(predictions)):
                combined_inputs[i][args['label_column_name']] = predictions[i]

        args['combined_inputs'] = combined_inputs

    if DATASET_TO_USE == "IMPLI":
        ## Only one phase!
        if not phase_2_generate:
            return loss_labels, None, outputs, predictions
        else:
            return generated_ids, predictions  ## Phase 1 preds

    args['phase_1'] = False

    ## phase_2_generate is true only during prediction
    ##     --> where we use beam search instead of using the output logits
    if phase_2_generate:
        model_inputs = _get_predictions_and_loss(args)  ## Overloaded with "phase_2_generate"
        generated_ids = args['model'].generate(
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            max_length=args['max_target_len'],
            use_cache=True,
            num_beams=NUM_BEAMS,
            early_stopping=True
        )

        return generated_ids, predictions  ## Phase 1 preds

    outputs, model_inputs = _get_predictions_and_loss(args)  ## For phase 2
    loss_expla = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    labels = model_inputs["labels"]

    return loss_labels, loss_expla, outputs, labels


@dataclass
class FigTrainingArguments(TrainingArguments):
    max_pred_len: int = field(default=False)
    max_src_len: int = field(default=False)
    max_target_len: int = field(default=False)
    prefix: str = field(default=False)
    prem_column_name: str = field(default=False)
    hyp_column_name: str = field(default=False)
    label_column_name: str = field(default=False)
    exp_column_name: str = field(default=False)
    padding: str = field(default=False)


class FigTrainer(Trainer):
    def compute_loss(self, model, combined_inputs, return_outputs=False):

        model.eval()
        loss_labels, loss_expla, outputs, model_inputs = _get_two_phase_output(
            combined_inputs=combined_inputs,
            prefix=self.args.prefix,
            prem_column_name=self.args.prem_column_name,
            hyp_column_name=self.args.hyp_column_name,
            label_column_name=self.args.label_column_name,
            exp_column_name=self.args.exp_column_name,
            tokenizer=self.tokenizer,
            max_pred_len=self.args.max_pred_len,
            max_src_len=self.args.max_src_len,
            max_target_len=self.args.max_target_len,
            padding=self.args.padding,
            device=self.args.device,
            model=model
        )

        model.train()
        loss = None
        if not loss_expla is None:
            loss = (1 - ALPHA) * loss_labels + ALPHA * loss_expla
        else:
            loss = loss_labels

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            combined_inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Overloaded.
        """

        model.eval()
        with torch.no_grad():
            loss_labels, loss_expla, outputs, labels = _get_two_phase_output(
                combined_inputs=combined_inputs,
                prefix=self.args.prefix,
                prem_column_name=self.args.prem_column_name,
                hyp_column_name=self.args.hyp_column_name,
                label_column_name=self.args.label_column_name,
                exp_column_name=self.args.exp_column_name,
                tokenizer=self.tokenizer,
                max_pred_len=self.args.max_pred_len,
                max_src_len=self.args.max_src_len,
                max_target_len=self.args.max_target_len,
                padding=self.args.padding,
                device=self.args.device,
                model=model
            )

        has_labels = True
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        if isinstance(outputs, dict):
            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
        else:
            logits = outputs[1:]

        loss = None
        if not loss_expla is None:
            loss = (1 - ALPHA) * loss_labels + ALPHA * loss_expla
        else:
            loss = loss_labels

        model.train()
        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        raise Exception("Todo: This is phase 1 labels, might need to return correct phase labels here")

        return (loss, logits, labels)


class T5FigurativeModel:
    def __init__(
            self,
            max_pred_len,
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
        self.max_pred_len = max_pred_len
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
            DATA_SOURCE,
            save_data=False,
            save_path=""
        )
        if DATASET_TO_USE == 'eSNLI':
            self.train_set, self.val_set, self.test_set = dataloader.get_processed_data()
        else:
            self.train_set, self.val_set = dataloader.get_processed_data()

    def _prepare_features(self, examples):

        if DATASET_TO_USE == 'eSNLI':
            if self.do_train:
                column_names = self.train_set.column_names
            elif self.do_eval:
                column_names = self.val_set.column_names
            elif self.do_predict:
                column_names = self.test_set.column_names
            else:
                raise exception("Unknown ... ")
        else:

            if self.do_train:
                column_names = self.train_set.column_names
            elif self.do_eval or self.do_predict:
                column_names = self.val_set.column_names

        # ['id', 'premise', 'hypothesis', 'label', 'explanation', 'split', 'type', 'idiom']
        self.prem_column_name = "premise" if "premise" in column_names else column_names[1]
        self.hyp_column_name = "hypothesis" if "hypothesis" in column_names else column_names[2]
        self.label_column_name = "label" if "label" in column_names else column_names[3]

        model_inputs = {
            'premises': examples[self.prem_column_name],
            'hypotheses': examples[self.hyp_column_name],
            'labels': examples[self.label_column_name],
        }

        self.exp_column_name = None
        if DATASET_TO_USE != "IMPLI":
            if DATASET_TO_USE == 'eFigSNLI':
                self.exp_column_name = "explanation" if "explanation" in column_names else column_names[4]
            elif DATASET_TO_USE == 'eSNLI':
                self.exp_column_name = "explanation_1" if "explanation_1" in column_names else column_names[4]
            else:
                raise Exception("Unknown dataset")

            model_inputs['explanations'] = examples[self.exp_column_name]

        return model_inputs

        # inputs, targets = self.preprocess_batch(
        #     examples,
        #     prem_column_name,
        #     hyp_column_name,
        #     label_column_name,
        #     exp_column_name
        # )

        # model_inputs = self.tokenizer(inputs, max_length=self.max_src_len, padding=self.padding, truncation=True)
        # # Setup the tokenizer for targets
        # with self.tokenizer.as_target_tokenizer():
        #     labels = self.tokenizer(targets, max_length=self.max_target_len, padding=self.padding, truncation=True)

        # # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # # padding in the loss.
        # if self.padding == "max_length":
        #     labels["input_ids"] = [
        #         [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #     ]

        # model_inputs["labels"] = labels["input_ids"]
        # return model_inputs

    def data_collator(self, features):
        return features
        # input_ids = [x["input_ids"] for x in features]
        # input_ids = torch.tensor(input_ids, dtype=torch.long)
        # attention_mask = [x["attention_mask"] for x in features]
        # attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        # labels = [x["labels"] for x in features]
        # labels = torch.tensor(labels, dtype=torch.long)

        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "labels": labels
        # }

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

        report_to = list()
        if WANDB:
            report_to = ["wandb"]

        args = FigTrainingArguments(
            max_pred_len=self.max_pred_len,
            max_src_len=self.max_src_len,
            max_target_len=self.max_target_len,
            prefix=self.prefix,
            prem_column_name=self.prem_column_name,
            hyp_column_name=self.hyp_column_name,
            label_column_name=self.label_column_name,
            exp_column_name=self.exp_column_name,
            padding=self.padding,
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            # evaluation_strategy="steps",
            # eval_steps=500,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            lr_scheduler_type=SCHEDULER,
            num_train_epochs=MAX_EPOCHS,
            logging_steps=10,
            save_steps=250,
            save_total_limit=2,
            run_name=RUN_NAME,
            disable_tqdm=False,
            report_to=report_to,
            remove_unused_columns=False,
            fp16=FP16,
            label_names=[
                "labels",
            ],  # it's important to log eval_loss
        )
        print("Batch Size", args.train_batch_size)
        print("Parallel Mode", args.parallel_mode)

        trainer = FigTrainer(
            model=self.model,
            args=args,
            data_collator=self.data_collator,
            train_dataset=train_dataset if self.do_train else None,
            eval_dataset=eval_dataset if self.do_eval else None,
            tokenizer=self.tokenizer
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
                data = eval_dataset[batch: batch + BATCH_SIZE]
                # print(data)
                prep_data = self.data_collator([data])
                model.eval()
                model.to(self.device)
                original_data = deepcopy(data)
                with torch.no_grad():
                    # https://huggingface.co/blog/how-to-generate

                    ## Updated based on two phase:
                    generated_ids, labels = _get_two_phase_output(
                        combined_inputs=data,
                        prefix=self.prefix,
                        prem_column_name=self.prem_column_name,
                        hyp_column_name=self.hyp_column_name,
                        label_column_name=self.label_column_name,
                        exp_column_name=self.exp_column_name,
                        tokenizer=self.tokenizer,
                        max_pred_len=self.max_pred_len,
                        max_src_len=self.max_src_len,
                        max_target_len=self.max_target_len,
                        padding=self.padding,
                        device=self.device,
                        model=model,
                        phase_2_generate=True
                    )

                # outputs = args[ 'tokenizer' ].batch_decode(generated_ids, skip_special_tokens=True)
                outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                if self.save_results:
                    explanations = None
                    if DATASET_TO_USE == 'IMPLI':
                        explanations = ["" for i in range(len(original_data["label"]))]
                        outputs = ["" for i in range(len(original_data["label"]))]
                    else:
                        explanations = original_data[self.exp_column_name]

                    for prem, hyp, label, exp, dec_preds, pred_label, in zip(
                            original_data["premise"],
                            original_data["hypothesis"],
                            original_data["label"],
                            explanations,
                            outputs,
                            labels
                    ):
                        # print(label)
                        # print(dec_preds[:1])
                        predictions.append({
                            "premise": prem,
                            "hypothesis": hyp,
                            "label": label,
                            "explanation": exp,
                            "predicted_label": pred_label,
                            "model_explanation": dec_preds.replace("explanation:", "").lstrip()
                        }
                        )
                    # print(predictions)
            with open(OUTPUT_DIR + "/outputs.json", "w") as f:
                f.write(json.dumps(predictions, indent=4))

        if WANDB:
            wandb.finish()


if __name__ == '__main__':

    ###
    # Training: python -m src.modelling.train_seq2seq_e2e
    # Predict on test Only (Set DO_TRAIN = False) and call as: python -m src.modelling.train_seq2seq_e2e <path to trained model>
    # Continue training from checkpoint: python -m src.modelling.train_seq2seq_e2e None <path to checkpoint to continue from>

    # DO_TRAIN = True
    # DO_EVAL = True
    # DO_PREDICT = True

    # DATASET_TO_USE = "IMPLI" ## IMPLI|eSNLI|eFigSNLI
    # DATASET_TO_USE = "eFigSNLI" ## IMPLI|eSNLI|eFigSNLI

    # # (alpha * explanation_loss) + (1-alpha)*label_loss, higher alpha is more importance to explanations
    # # Will be ignored if DATASET_TO_USE == "IMPLI"
    # ALPHA = 0.9

    # RUN_NAME   = "t5-e2e-1epochs-lr1e4-IMPLI-A0-9"
    # OUTPUT_DIR = "/content/drive/MyDrive/TMP/" + RUN_NAME
    # IMPLI_DATA_LOCATION = "/content/UKP-FigLang2022/data/impli/"

    import argparse

    parser = argparse.ArgumentParser(prog='train_seq2seq_e2e')
    parser.add_argument('--resume_training', help='Location of model to continue training from', default=None)
    parser.add_argument('--checkpoint',
                        help='Location of model to run predictions on (must set do_train and do_eval to False)',
                        default=None)

    parser.add_argument('--do_train', action='store_true', help='do_train (default)')
    parser.add_argument('--no_do_train', dest='do_train', action='store_false', help='do not do_train')
    parser.set_defaults(do_train=True)

    parser.add_argument('--do_eval', action='store_true', help='do_eval (default)')
    parser.add_argument('--no_do_eval', dest='do_eval', action='store_false', help='do not do_eval')
    parser.set_defaults(do_eval=True)

    parser.add_argument('--do_predict', action='store_true', help='do_predict (default)')
    parser.add_argument('--no_do_predict', dest='do_predict', action='store_false', help='do not do_predict')
    parser.set_defaults(do_predict=True)

    parser.add_argument('--dataset_to_use', help='IMPLI|eSNLI|eFigSNLI (default=eFigSNLI)')
    parser.set_defaults(data_set_to_use='eFigSNLI')

    parser.add_argument('--impli_data_location', help='Full path to IMPLI data', default=None)
    parser.add_argument('--max_epochs', help='max_epochs (default 10)', type=int)
    parser.set_defaults(max_epochs=10)

    parser.add_argument('--model_name', help='t5-base (default) or t5-large', default='t5-base')

    parser.add_argument('--efigsnli_data_source', help='Local location of ColumbiaNLP/FigLang2022SharedTask',
                        default=None)

    parser.add_argument('--output_dir', help='output_dir: = output_dir + run_name', required=True)
    parser.add_argument('--run_name', help='run name', required=True)
    parser.add_argument('--alpha',
                        help='Between 0 and 1: (alpha * explanation_loss) + (1-alpha)*label_loss, higher alpha is more importance to explanations',
                        required=True, type=float)

    args = parser.parse_args()

    if not args.resume_training is None:
        RESUME_TRAINING = args.resume_training

    if not args.checkpoint is None:
        CHECKPOINT = args.checkpoint

    if not args.impli_data_location is None:
        IMPLI_DATA_LOCATION = args.impli_data_location

    if not args.efigsnli_data_source is None:
        eFigSNLI_DATA_SOURCE = args.efigsnli_data_source

    DO_TRAIN = args.do_train
    DO_EVAL = args.do_eval
    DO_PREDICT = args.do_predict

    DATASET_TO_USE = args.dataset_to_use
    MAX_EPOCHS = args.max_epochs
    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir
    RUN_NAME = args.run_name
    ALPHA = args.alpha

    OUTPUT_DIR = os.path.join(OUTPUT_DIR, RUN_NAME)

    if DATASET_TO_USE == "eFigSNLI":
        from src.e2e.dataloader import PreprocessData

        DATA_SOURCE = eFigSNLI_DATA_SOURCE
    elif DATASET_TO_USE == "IMPLI":
        from src.e2e.impli_dataloader import PreprocessData

        DATA_SOURCE = IMPLI_DATA_LOCATION
    elif DATASET_TO_USE == "eSNLI":
        from src.e2e.esnli_dataloader import PreprocessData

        DATA_SOURCE = "esnli"
    else:
        raise Exception("Unknown data source, use one of eFigSNLI|IMPLI|eSNLI")

    print("""
####################     PARAMETERS:      ####################
WANDB: """ + str(WANDB) + """
USE_PREDICTED_LABELS: """ + str(USE_PREDICTED_LABELS) + """
NUM_BEAMS: """ + str(NUM_BEAMS) + """
MAX_TRAIN_SAMPLES: """ + str(MAX_TRAIN_SAMPLES) + """
MAX_VAL_SAMPLES: """ + str(MAX_VAL_SAMPLES) + """
LEARNING_RATE: """ + str(LEARNING_RATE) + """
WARMUP_STEPS: """ + str(WARMUP_STEPS) + """
DROPOUT: """ + str(DROPOUT) + """
FP16: """ + str(FP16) + """
SCHEDULER: """ + str(SCHEDULER) + """
BATCH_SIZE: """ + str(BATCH_SIZE) + """
DO_TRAIN: """ + str(DO_TRAIN) + """
DO_EVAL: """ + str(DO_EVAL) + """
DO_PREDICT: """ + str(DO_PREDICT) + """
MAX_EPOCHS: """ + str(MAX_EPOCHS) + """ 
DATASET_TO_USE: """ + str(DATASET_TO_USE) + """
ALPHA: """ + str(ALPHA) + """
RUN_NAME: """ + str(RUN_NAME) + """
OUTPUT_DIR: """ + str(OUTPUT_DIR) + """
IMPLI_DATA_LOCATION: """ + str(IMPLI_DATA_LOCATION) + """
MODEL_NAME: """ + str(MODEL_NAME) + """
RESUME_TRAINING: """ + str(RESUME_TRAINING) + """
CHECKPOINT: """ + str(CHECKPOINT) + """
DATA_SOURCE: """ + str(DATA_SOURCE) + """
eFigSNLI_DATA_SOURCE: """ + str(eFigSNLI_DATA_SOURCE) + """ 
PreprocessData: """ + str(PreprocessData) + """
    """)

    trainer = T5FigurativeModel(
        max_pred_len=6,
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
