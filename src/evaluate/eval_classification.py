import pandas as pd
import numpy as np

outputs = "t5-impli-10epochs-lr1e4-esnli-figlang/outputs.json"
# outputs_with_labels = "t5-fig-lang-10epochs-lr1e4-gold/outputs.json"

with open(outputs, "r") as file1:
    data = file1.read()

df = pd.read_json(data)
# print("Mean w/o label information:", df["explanationscore"].mean())
print(df.head())

# with open(outputs_with_labels, "r") as file2:
#     data = file2.read()
#
# df = pd.read_json(data)
# print("Mean with label information:", df["explanationscore"].mean())

df['result'] = np.where(df['label'] == df['predicted_label'], df['predicted_label'], np.nan)
print(df["result"])
print(df['result'].count()/df.shape[0])
