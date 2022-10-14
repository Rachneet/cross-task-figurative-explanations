import json

with open("t5-large-figlang-label/outputs_final_task.json", "r") as f:
    data = json.load(f)

    for item in data:
        with open("submissions/answer.json", "a") as file:
            file.write(json.dumps(item))
            file.write('\n')

    file.close()
