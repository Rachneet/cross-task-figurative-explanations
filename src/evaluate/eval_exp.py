from bert_score import score
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

filename = "t5-fig-lang-10epochs-lr1e4-gold/outputs.json"

with open(filename) as f:
    data = json.load(f)

cands1 = []
refs = []
for i in range(len(data)):
    if "explanationscore" not in data[i]:
        cands1.append(data[i]["model_explanation"])
        refs.append(data[i]["explanation"])
        print(data[i])

P1, R1, F1 = score(
    cands1,
    refs,
    lang="en",
    model_type='microsoft/deberta-large-mnli',
    batch_size=1,
    device="cuda:0"
)
F1 = F1.cpu().detach().numpy().tolist()


from bleurt import score
scorer = score.BleurtScorer('BLEURT-20')
BLEURTscores = scorer.score(references=refs, candidates=cands1, batch_size=1)


for i in range(len(data)):
    if "explanationscore" not in data[i]:
        cands1.append(data[i]["model_explanation"])
        refs.append(data[i]["explanation"])
        data[i]["explanationscore"] = int((F1[i]+BLEURTscores[i])*50.0)


with open(filename, "w") as f:
    f.write(json.dumps(data, indent=4)+'\n')
