import json
import pandas as pd
from tqdm import tqdm

input_file = "/data/zez/Boundary/temp/guardreasoner-filtered.jsonl"
# input_file = "/data/zez/Boundary/temp/guardreasoner-sample.jsonl"
output_file = "/data/zez/Boundary/train/tail_confidence_by_percent-top_10K_samples.parquet"
# output_file = "/data/zez/Boundary/train/random_sample-10K_samples.parquet"


print(f"[INFO] Loading jsonl file from {input_file}")
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

train_set = []
for item in tqdm(data, desc="Formating Data"):
    input_key = item["conversations"][0]["content"]
    output_key = item["conversations"][1]["content"]

    train_set.append({
        "question": input_key,
        "response": output_key
    })

print(f"[INFO] Saving training parquet file to {output_file}")
df = pd.DataFrame(train_set)
df.to_parquet(output_file)
