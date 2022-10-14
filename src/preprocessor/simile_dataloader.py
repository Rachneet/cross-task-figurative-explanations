import itertools
from pprint import pprint
import os
import json
from tqdm import tqdm

from collections import OrderedDict, Counter
from datasets import load_dataset, load_from_disk


# data processing for the simile dataset
# the premise is the literal sentence
# and the hypothesis is the figurative simile sentence
# we want to generate similes


class PreprocessData:
    def __init__(self,
                 dataset_path: str,
                 save_data: bool = False,
                 save_path: str = ""
                 ):
        """
        load datasets
        """
        self.train_set = load_from_disk(dataset_path+"train")
        self.val_set = load_from_disk(dataset_path+"test")
        self.save_data = save_data
        self.save_path = save_path

    def get_data_info(self):
        """
        data statistics
        """
        train_samples = self.train_set.shape[0]
        val_samples = self.val_set.shape[0]
        column_names = self.val_set.column_names
        train_ex = self.train_set.select([0])[0]
        val_ex = self.val_set.select([0])[0]
        info = {
            "train_samples": train_samples,
            "val_samples": val_samples,
            "column_names": column_names,
            "train_ex": train_ex,
            "val_ex": val_ex
        }
        return OrderedDict(info)

    def percent_counts(self, input: dict):
        """
        Get the percent value counts in dataset column
        :param input:
        :return:
        """
        percent = [(k, round(v/sum(list(input.values())), 2)) for k, v in input.items()]
        return percent

    def value_counts(self, train_data, val_data, column):
        """
        get the counts of unique values in dataset columns
        :param train_data:
        :param val_data:
        :param column:
        :return:
        """
        tr_counts = dict(Counter(train_data[column]))
        val_counts = dict(Counter(val_data[column]))
        tr_percent_counts = self.percent_counts(tr_counts)
        val_percent_counts = self.percent_counts(val_counts)
        return tr_percent_counts, val_percent_counts

    def get_sentence_length(self, train_data, val_data, column):
        tr_len = [len(sent.split(" "))for sent in train_data[column]]
        tr_avg_count = sum(tr_len)/len(tr_len)
        val_len = [len(sent.split(" ")) for sent in val_data[column]]
        val_avg_count = sum(val_len) / len(val_len)
        tr_max_len, val_max_len = max(tr_len), max(val_len)
        return tr_max_len, val_max_len, round(tr_avg_count, 2), round(val_avg_count, 2)

    def get_train_val_stats(self):
        """
        Get the statistics for the train and test sets
        :return:
        """
        # avg. word lengths
        len_premise = self.get_sentence_length(
            self.train_set,
            self.val_set,
            column="premise")
        len_hypothesis = self.get_sentence_length(
            self.train_set,
            self.val_set,
            column="hypothesis")

        return len_premise, len_hypothesis

    def process_samples(self, example):
        example["hypothesis"] = example["hypothesis"].replace("<EOT>", "").rstrip()
        return example

    def get_processed_data(self):
        proc_train_set = self.train_set.map(self.process_samples)
        proc_val_set = self.val_set.map(self.process_samples)
        return proc_train_set, proc_val_set


if __name__ == '__main__':
    save_dir = ""
    dataloader = PreprocessData("../../data/simile/", save_data=False, save_path=save_dir)
    print(dataloader.get_train_val_stats())
    t, v = dataloader.get_processed_data()
    print(t[0])
    print(v[0])
