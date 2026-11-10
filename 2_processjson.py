import json
import csv
from pathlib import Path

base_dir = Path(__file__).resolve().parent
INPUT_DIR = base_dir / "json"
OUTPUT_CSV = base_dir / "llm_answers.csv"

rows = []
all_question_keys = set()

def parse_content(data):
    content = data.get("content")

    if isinstance(content, dict):
        return content

    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None

for json_file in INPUT_DIR.glob("*.json"):
    if json_file.name == "_run_summary.json":
        print(f"Skipping log file: {json_file.name}")
        continue

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        pid = str(data.get("pid", ""))
        content_dict = parse_content(data)

        if content_dict is None:
            print(f"Skipping {json_file.name}: content could not be parsed")
            continue

        row = {"id": pid}
        for k, v in content_dict.items():
            k = str(k)
            row[k] = v
            all_question_keys.add(k)

        rows.append(row)

    except Exception as e:
        print(f"Failed to process {json_file.name}: {e}")

def sort_mixed_key(x):
    try:
        return (0, int(x))
    except:
        return (1, x)

question_keys = sorted(all_question_keys, key=sort_mixed_key)

def sort_id(row):
    try:
        return int(row["id"])
    except:
        return row["id"]

rows = sorted(rows, key=sort_id)

fieldnames = ["id"] + question_keys

with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to: {OUTPUT_CSV}")
print(f"Rows written: {len(rows)}")
print("Columns:", fieldnames)
