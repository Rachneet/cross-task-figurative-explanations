import datasets
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict, load_from_disk

idioms_e_df = pd.read_csv(
    "../../data/impli_raw/manual_idioms_e.tsv",
    sep="\t",
    names=["premise", "hypothesis"]
)
idioms_e_df["label"] = ["Entailment"] * idioms_e_df.shape[0]
# print(idioms_e_df.head())
# print(idioms_e_df.shape)   # 528


idioms_ne_df = pd.read_csv(
    "../../data/impli_raw/manual_idioms_ne.tsv",
    sep="\t",
    names=["premise", "hypothesis"]
)
idioms_ne_df["label"] = ["Contradiction"] * idioms_ne_df.shape[0]
# print(idioms_ne_df.head())
# print(idioms_ne_df.shape)   # 254

# idioms_semeval_e_df = pd.read_csv(
#     "../../data/impli_raw/fig_context_semeval_e.tsv",
#     sep="\t",
#     names=["premise", "hypothesis"]
# )
# idioms_semeval_e_df["label"] = ["Entailment"] * idioms_semeval_e_df.shape[0]
# # print(idioms_semeval_e_df.head())
# # print(idioms_semeval_e_df.shape)   # 587
#
#
# idioms_ant_ne_df = pd.read_csv(
#     "../../data/impli_raw/manual_antonyms_ne.tsv",
#     sep="\t",
#     names=["premise", "hypothesis"]
# )
# idioms_ant_ne_df["label"] = ["Contradiction"] * idioms_ant_ne_df.shape[0]
# # print(idioms_ant_ne_df.head())
# # print(idioms_ant_ne_df.shape)   # 375


metaphors_e_df = pd.read_csv(
    "../../data/impli_raw/manual_metaphors_e.tsv",
    sep="\t",
    names=["premise", "hypothesis"]
)
metaphors_e_df["label"] = ["Entailment"] * metaphors_e_df.shape[0]
# print(meatphors_e_df.head())
# print(meatphors_e_df.shape)   # 387


metaphors_ne_df = pd.read_csv(
    "../../data/impli_raw/manual_metaphors_ne.tsv",
    sep="\t",
    names=["premise", "hypothesis"]
)
metaphors_ne_df["label"] = ["Contradiction"] * metaphors_ne_df.shape[0]
# print(metaphors_ne_df.head())
# print(metaphors_ne_df.shape)   # 281

# metaphors_repl_cc_e_df = pd.read_csv(
#     "../../data/impli_raw/replacement_cc_e.tsv",
#     sep="\t",
#     names=["premise", "hypothesis"]
# )
# metaphors_repl_cc_e_df["label"] = ["Entailment"] * metaphors_repl_cc_e_df.shape[0]
# # print(metaphors_repl_cc_e_df.head())
# # print(metaphors_repl_cc_e_df.shape)   # 545
#
#
# metaphors_repl_tsv_e_df = pd.read_csv(
#     "../../data/impli_raw/replacement_tsvetkov_e.tsv",
#     sep="\t",
#     names=["premise", "hypothesis"],
#     encoding="ISO-8859-1"
# )
# metaphors_repl_tsv_e_df["label"] = ["Entailment"] * metaphors_repl_tsv_e_df.shape[0]
# # print(metaphors_repl_tsv_e_df.head())
# # print(metaphors_repl_tsv_e_df.shape)  # 100


frames = [idioms_e_df, idioms_ne_df, metaphors_e_df, metaphors_ne_df]
result = pd.concat(frames)
print(result.shape)

# remove empty samples
nan_value = float("NaN")
result.replace("", nan_value, inplace=True)
result.dropna(subset=["hypothesis"], inplace=True)
result = result.reset_index(drop=True)
result = result.sample(frac=1).reset_index(drop=True)

print(result.head(10))
print(result.shape)

dataset = Dataset.from_pandas(result)
# dataset.save_to_disk(dataset_path="../../data/impli.hf")
print(dataset)

# # 90% train, 10% test + validation
# train_test_valid = dataset.train_test_split(test_size=0)
# # Split the 10% test + valid in half test, half valid
# # test_valid = train_test_valid['test'].train_test_split(test_size=0.5)
# # gather everyone if you want to have a single DatasetDict
# train_test_valid_dataset = DatasetDict({
#     'train': dataset['train'],
#     # 'valid': test_valid['train']
# })
#
dataset.save_to_disk("../../data/impli_full")
# from datasets import load_dataset

# dataset = load_dataset("UKP/impli")
# print(dataset)
