import sys

from collections import OrderedDict, Counter
from datasets import load_dataset


# Data loading and processing for the e-SNLI dataset, downloaded from HuggingFace.
# This will load all three gold explanation columns
# Later on we can choose to discard the gold explanations we don't want


class PreprocessData:
    def __init__(self,
                 dataset_name: str,
                 save_data: bool = False,
                 convert_labels: bool = True,
                 save_path: str = ""
                 ):
        """
        load datasets
        """
        self.train_set = load_dataset(dataset_name, split="train")
        self.val_set = load_dataset(dataset_name, split="validation")
        self.test_set = load_dataset(dataset_name, split="test")
        self.save_data = save_data
        self.save_path = save_path
        self.convert_labels = convert_labels

    def get_data_info(self):
        """
        data statistics
        """
        train_samples = self.train_set.shape[0]
        val_samples = self.val_set.shape[0]
        test_samples = self.test_set.shape[0]
        column_names = self.test_set.column_names
        train_ex = self.train_set.select([0])[0]
        val_ex = self.val_set.select([0])[0]
        test_ex = self.test_set.select([0])[0]
        info = {
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "column_names": column_names,
            "train_ex": train_ex,
            "val_ex": val_ex,
            "test_ex": test_ex,
        }
        return OrderedDict(info)

    def percent_counts(self, input: dict):
        """
        Get the percent value counts in dataset column
        :param input:
        :return:
        """
        percent = [(k, round(v / sum(list(input.values())), 2)) for k, v in input.items()]
        return percent

    def value_counts(self, train_data, val_data, test_data, column):
        """
        get the counts of unique values in dataset columns
        :param train_data:
        :param val_data:
        :param column:
        :return:
        """
        tr_counts = dict(Counter(train_data[column]))
        val_counts = dict(Counter(val_data[column]))
        test_counts = dict(Counter(test_data[column]))
        tr_percent_counts = self.percent_counts(tr_counts)
        val_percent_counts = self.percent_counts(val_counts)
        test_percent_counts = self.percent_counts(test_counts)
        return tr_percent_counts, val_percent_counts, test_percent_counts

    def get_sentence_length(self, train_data, val_data, test_data, column):
        tr_len = [len(sent.split(" ")) for sent in train_data[column]]
        tr_avg_count = sum(tr_len) / len(tr_len)
        val_len = [len(sent.split(" ")) for sent in val_data[column]]
        val_avg_count = sum(val_len) / len(val_len)
        test_len = [len(sent.split(" ")) for sent in test_data[column]]
        test_avg_count = sum(test_len) / len(test_len)
        tr_max_len, val_max_len, test_max_len = max(tr_len), max(val_len), max(test_len)
        return tr_max_len, val_max_len, test_max_len, \
               round(tr_avg_count, 2), round(val_avg_count, 2), \
               round(test_avg_count, 2)

    def get_train_val_stats(self):
        """
        Get the statistics for the train and test sets
        :return:
        """
        # label counts
        tr_labels, val_labels, test_labels = self.value_counts(
            self.train_set,
            self.val_set,
            self.test_set,
            column="label")

        # avg. sentence lengths
        len_premise = self.get_sentence_length(
            self.train_set,
            self.val_set,
            self.test_set,
            column="premise")
        len_hypothesis = self.get_sentence_length(
            self.train_set,
            self.val_set,
            self.test_set,
            column="hypothesis")
        len_exp1 = self.get_sentence_length(
            self.train_set,
            self.val_set,
            self.test_set,
            column="explanation_1")
        len_exp2 = self.get_sentence_length(
            self.train_set,
            self.val_set,
            self.test_set,
            column="explanation_2")
        len_exp3 = self.get_sentence_length(
            self.train_set,
            self.val_set,
            self.test_set,
            column="explanation_3")

        return tr_labels, val_labels, len_premise, \
               len_hypothesis, len_exp1, len_exp2, len_exp3

    def get_processed_data(self):
        # remove neutral label
        proc_train_set = self.train_set.filter(lambda example: example["label"] != 1)
        proc_val_set = self.val_set.filter(lambda example: example["label"] != 1)
        proc_test_set = self.test_set.filter(lambda example: example["label"] != 1)

        return proc_train_set, proc_val_set, proc_test_set


if __name__ == '__main__':

    if len(sys.argv) > 1:
        save_dir = sys.arv[1]
    else:
        save_dir = "/data/e-snli"

    dataset = PreprocessData("esnli", save_data=False, save_path=save_dir)
    train, val, test = dataset.get_processed_data()
    print(train[0])
    print(val[0])
    print(test[0])
    # print(dataset.get_train_val_stats())
