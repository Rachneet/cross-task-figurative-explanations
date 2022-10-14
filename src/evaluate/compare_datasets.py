from collections import Counter
import re
import string
import pandas as pd
import ast
from tqdm import tqdm

from src.preprocessor import dataloader, impli_dataloader


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Compute f1 score of prediction
    :param prediction:
    :param ground_truth:
    :return:
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    Exact match for Squad evaluation
    :param prediction:
    :param ground_truth:
    :return:
    """
    if normalize_answer(prediction) == normalize_answer(ground_truth):
        return 1
    else:
        return 0


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_score(metric, pred_text, gold_text):
    # print(type(pred_text[0][0]["answer"]))
    # pred_text = pred_text[0][0]["answer"]
    # print(gold_text)
    if metric == "exact_match":
        score = metric_max_over_ground_truths(exact_match_score,
                                              pred_text,
                                              gold_text)
    elif metric == "f1":
        score = metric_max_over_ground_truths(f1_score, pred_text, gold_text)
    return score


if __name__ == '__main__':

    figlang = dataloader.PreprocessData(
        "ColumbiaNLP/FigLang2022SharedTask", save_data=False, save_path="")
    impli = impli_dataloader.PreprocessData(
        "../../data/impli_new/", save_data=False, save_path="")
    fig_train, fig_val = figlang.get_processed_data()
    fig_full_data = fig_train["premise"] + fig_val["premise"]

    impli_train, impli_val = impli.get_processed_data()
    impli_full_data = impli_train["premise"] + impli_val["premise"]

    print(len(impli_full_data))
    print(len(fig_full_data))
    # chunk into equal sizes
    chunk1 = fig_full_data[: len(impli_full_data)]
    chunk2 = fig_full_data[len(impli_full_data): len(impli_full_data)*2]
    chunk3 = fig_full_data[len(impli_full_data)*2: len(impli_full_data)*3]
    chunk4 = fig_full_data[len(impli_full_data) * 3:] + \
             (["null"]*(len(impli_full_data) - (len(fig_full_data) - len(impli_full_data)*3)))
    chunks = [chunk1, chunk2, chunk3, chunk4]

    print(len(chunk1))
    print(len(chunk2))
    print(len(chunk3))
    print(len(chunk4))

    exact_match = []
    for chunk in chunks:
        for prem, prem2 in tqdm(zip(impli_full_data, chunk)):
            exact_match.append(get_score("exact_match", prem, prem2))

    print(exact_match)
    print(sum(exact_match))
