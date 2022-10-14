import json
import ast
import pandas as pd
import numpy as np


with open("../../submissions/sub2_esnli_imp_td.json", "r") as f:
    data_old = f.readlines()
    data_old = [ast.literal_eval(d) for d in data_old]
    df = pd.DataFrame(data_old)

with open("../../submissions/e_td_label.json", "r") as f:
    data_new = f.readlines()
    data_new = [ast.literal_eval(d) for d in data_new]
    # print(data_new)
    # print(data_new["predicted_label"])
    df["new_label"] = [item["predicted_label"] for item in data_new]

# print(df.head())
# print(df.shape)
df = df[df['new_label'] != df['predicted_label']]
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', -1)
df1 = df[["premise", "hypothesis", "new_label", "predicted_label"]]
print(df1.shape)
print(df1.tail(20))
