import datasets
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict, load_from_disk

# with open("../../data/simile/train.source", 'r') as f:
#     data = f.read().replace('\n', ',').replace(',,', '\n')

train_source = pd.read_fwf(
    "../../data/simile/train.source",
    encoding="utf-8",
    names=["premise", "remove_1", "remove_2"]
)
train_source.drop(["remove_1", "remove_2"], axis=1, inplace=True)

print(train_source.head())
print(train_source.shape)

train_target = pd.read_fwf(
    "../../data/simile/train.target",
    encoding="utf-8",
    names=["hypothesis", "end"]
)

train_target.drop(["end"], axis=1, inplace=True)
print(train_target.head())
print(train_source.shape)

val_source = pd.read_fwf(
    "../../data/simile/val.source",
    encoding="utf-8",
    names=["premise"]
)
# train_source.drop(["remove_1", "remove_2"], axis=1, inplace=True)

print(val_source.head())
print(val_source.shape)

val_target = pd.read_fwf(
    "../../data/simile/val.target",
    encoding="utf-8",
    names=["hypothesis", "end_1", "end_2"]
)
val_target.drop(["end_1", "end_2"], axis=1, inplace=True)

print(val_target.head())
print(val_target.shape)

train_data = pd.concat([train_source, train_target], axis=1, names=["premise", "hypothesis"])
with pd.option_context('display.max_columns', None, 'expand_frame_repr', False):
    print(train_data.head())

print(train_data.shape)  # (82697, 2)

val_frames = [val_source, val_target]
val_data = pd.concat(val_frames, axis=1, names=["premise", "hypothesis"])
print(val_data.shape)   # (5146, 2)

# remove empty samples
nan_value = float("NaN")
train_data.replace("", nan_value, inplace=True)
train_data.dropna(subset=["premise", "hypothesis"], inplace=True)
train_data = train_data.reset_index(drop=True)
train_data = train_data.sample(frac=1).reset_index(drop=True)

print(train_data.head(10))
print(train_data.shape)

val_data.replace("", nan_value, inplace=True)
val_data.dropna(subset=["premise", "hypothesis"], inplace=True)
val_data = val_data.reset_index(drop=True)
val_data = val_data.sample(frac=1).reset_index(drop=True)


print(val_data.head(10))
print(val_data.shape)

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
print(train_dataset)
print(val_dataset)

dataset = DatasetDict({
    'train': train_dataset,
    'test': val_dataset,
})

print(dataset)

dataset.save_to_disk("../../data/simile")
