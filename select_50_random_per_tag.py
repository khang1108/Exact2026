import csv
import re
import random

def select_50_per_tag():
    input_file = r'd:\EXACT_2026\src\exact\datasets\exact\type2_physics_questions.csv'
    output_file = r'd:\EXACT_2026\src\exact\datasets\exact\type2_physics_questions_50_per_tag.csv'

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by ID prefix (letters only)
    categories = {}
    for row in rows:
        q_id = row['id']
        match = re.match(r'^([a-zA-Z]+)', q_id)
        if match:
            cat = match.group(1)
        else:
            cat = 'OTHER'
        
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(row)

    selected_rows = []
    random.seed(42)  # For reproducible random sampling

    # Select up to 50 random questions per tag
    for cat, items in categories.items():
        if len(items) <= 50:
            selected_items = items
        else:
            selected_items = random.sample(items, 50)
        selected_rows.extend(selected_items)

    # Write to CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        if selected_rows:
            writer = csv.DictWriter(f, fieldnames=selected_rows[0].keys())
            writer.writeheader()
            writer.writerows(selected_rows)

    print(f"Selected {len(selected_rows)} questions across {len(categories)} categories.")
    for cat, items in categories.items():
        count = min(len(items), 50)
        print(f"  {cat}: {len(items)} total -> selected {count}")

if __name__ == '__main__':
    select_50_per_tag()
