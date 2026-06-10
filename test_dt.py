import json
import os
import sys

# Ensure correct path
sys.path.insert(0, r"d:\EXACT_2026\src")

from exact.config import get_settings
from exact.common.schemas import PredictionRequest
from exact.type2.pipeline import run_type2_pipeline

def main():
    settings = get_settings()
    
    with open(r'd:\EXACT_2026\result\v14_DT\kaggle_full_370.json', encoding='utf8') as f:
        data = json.load(f).get('predictions', [])
        
    dt_rows = [r for r in data if 'DT' in r['id']]
    
    for row in dt_rows[:3]:
        req = PredictionRequest(id=row['id'], question=row['question'])
        try:
            res = run_type2_pipeline(req, settings=settings)
            print(f"{row['id']}: {res.answer} {res.unit} (gold: {row.get('gold')})")
            if res.routing_diagnostics:
                print(f"  route: {res.routing_diagnostics.get('predicted_method')}")
        except Exception as e:
            import traceback
            print(f"{row['id']}: ERROR {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
