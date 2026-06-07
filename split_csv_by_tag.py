import csv
import re
import os
from collections import defaultdict

input_file = r"D:\EXACT_2026\src\exact\datasets\exact\type2_physics_questions.csv"
output_dir = os.path.dirname(input_file)

data_by_tag = defaultdict(list)
header = []

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if not row: continue
        id_val = row[0]
        match = re.match(r'^([A-Za-z]+)', id_val)
        if match:
            tag = match.group(1)
            data_by_tag[tag].append(row)
        else:
            data_by_tag['unknown'].append(row)

for tag, rows in data_by_tag.items():
    output_path = os.path.join(output_dir, f"type2_physics_questions_{tag}.csv")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Created {output_path} with {len(rows)} rows.")
