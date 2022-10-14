from bert_score import score
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

filename = "t5-large-figlang/outputs_10beams.json"

with open(filename) as f:
    data = json.load(f)

count = 0

cands1 = []
refs = []
for i in range(len(data)):
    if data[i]["predicted_label"] != "" and "explanationscore" not in data[i]:
        cands1.append(data[i]["model_explanation"])
        refs.append(data[i]["explanation"])
        print(data[i])

P1, R1, F1 = score(
    cands1,
    refs,
    lang="en",
    model_type="microsoft/deberta-large-mnli",
    batch_size=1,
    device="cuda:0",
)
F1 = F1.cpu().detach().numpy().tolist()


from bleurt import score

scorer = score.BleurtScorer("BLEURT-20")
BLEURTscores = scorer.score(references=refs, candidates=cands1, batch_size=1)


for i in range(len(data)):
    if (
        data[i]["predicted_label"] == data[i]["label"]
        and "explanationscore" not in data[i]
    ):
        cands1.append(data[i]["model_explanation"])
        refs.append(data[i]["explanation"])
        data[i]["explanationscore"] = int((F1[i] + BLEURTscores[i]) * 50.0)


with open(filename, "w") as f:
    f.write(json.dumps(data, indent=4) + "\n")

with open(filename) as f:
    data = json.load(f)

count = 0
count1 = 0
count2 = 0
for line in data:
    if line["label"] == line["predicted_label"] and line["explanationscore"] >= 0:
        count = count + 1
    if line["label"] == line["predicted_label"] and line["explanationscore"] >= 50:
        count1 = count1 + 1
    if line["label"] == line["predicted_label"] and line["explanationscore"] >= 60:
        count2 = count2 + 1

print("Accuracy@0", count / len(data))
print("Accuracy@50", len(data), count1 / len(data))
print("Accuracy@60", len(data), count2 / len(data))
