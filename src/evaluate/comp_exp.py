import pandas as pd

outputs_wo_labels = "t5-fig-lang-10epochs-lr1e4-prefix/outputs_lp.json"
outputs_with_labels = "t5-fig-lang-10epochs-lr1e4-gold/outputs.json"

with open(outputs_wo_labels, "r") as file1:
    data = file1.read()

df = pd.read_json(data)
print("Mean w/o label information:", df["explanationscore"].mean())


with open(outputs_with_labels, "r") as file2:
    data = file2.read()

df = pd.read_json(data)
print("Mean with label information:", df["explanationscore"].mean())
