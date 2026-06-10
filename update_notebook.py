import json

def update_notebook():
    file_path = r'd:\EXACT_2026\notebooks\type2_kaggle_benchmark.ipynb'

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace the CSV filename
    content = content.replace('src/exact/datasets/exact/type2_physics_questions.csv', 'src/exact/datasets/exact/type2_physics_questions_50_per_tag.csv')
    
    # Replace the hardcoded limit and output file names
    content = content.replace('1352', '370')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Notebook updated successfully.")

if __name__ == '__main__':
    update_notebook()
