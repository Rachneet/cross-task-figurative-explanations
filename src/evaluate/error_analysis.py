import pandas as pd
from datasets import load_dataset
import numpy as np


predictions = "../../results/outputs_fig_impli.json"

# load predictions
df = pd.read_json(predictions)

# load figurative type
val_set = load_dataset("ColumbiaNLP/FigLang2022SharedTask", split="train[:10%]")
df["type"] = val_set["type"]


df['result'] = np.where(df['label'] == df['predicted_label'], df['predicted_label'], np.nan)
# print(df.head(20))

type_counts = df.groupby(['type']).size().reset_index(name='counts')
print(type_counts.head())

data = df.groupby(['type', 'result']).describe()
print(data)
