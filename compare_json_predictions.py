import json
import sys

files = {
    'DT': r'D:\EXACT_2026\result\v15_DT2\kaggle_full_370.json',
    'NL': r'D:\EXACT_2026\result\v16_NL2\kaggle_full_370.json',
    'LD': r'D:\EXACT_2026\result\v17_LD2\kaggle_full_370.json'
}

data = {}
for name, path in files.items():
    with open(path, 'r', encoding='utf-8') as f:
        data[name] = json.load(f)

print("--- SUMMARIES ---")
for name in ['DT', 'NL', 'LD']:
    summary = data[name].get('summary', {})
    acc = summary.get('accuracy', 0)
    total = summary.get('total', 0)
    correct = summary.get('correct', 0)
    pipeline_errors = summary.get('pipeline_errors', 0)
    print(f"{name}: Accuracy = {acc:.4f} ({correct}/{total}), Pipeline Errors = {pipeline_errors}")

print("\n--- DETAILED COMPARISON ---")
preds = {}
for name in ['DT', 'NL', 'LD']:
    preds[name] = {}
    for p in data[name]['predictions']:
        # Extract numeric part of ID, e.g., physics_NL001 -> 001
        qid = ''.join(filter(str.isdigit, p['id']))
        preds[name][qid] = p

ids = set(preds['NL'].keys())

# Find where NL is correct, but DT is wrong
nl_correct_dt_wrong = []
nl_correct_ld_wrong = []
dt_errors = {}
ld_errors = {}

for qid in ids:
    p_nl = preds['NL'].get(qid, {})
    p_dt = preds['DT'].get(qid, {})
    p_ld = preds['LD'].get(qid, {})
    
    # We can check correctness by routing_log['correct'] or just checking if 'correct' is in routing_log
    def is_correct(p):
        log = p.get('routing_log') or {}
        # some formats might just check if error is null and gold_answer == answer
        # The file format has summary['correct'], so there must be a way. Let's just use answer matching or log
        if p.get('routing_log') and p['routing_log'].get('correct'):
            return True
        if p.get('gold_answer') is not None and str(p.get('answer')).strip() == str(p.get('gold_answer')).strip():
            return True
        # Check numeric equivalence? We can rely on summary's numbers or just look at pipeline errors vs wrong.
        # But actually let's just look at 'error' field and 'routing_diagnostics'
        return (p.get('routing_log') or {}).get('correct', False)

    nl_c = is_correct(p_nl)
    dt_c = is_correct(p_dt)
    ld_c = is_correct(p_ld)

    if nl_c and not dt_c:
        nl_correct_dt_wrong.append(qid)
        err = p_dt.get('error')
        if err:
            dt_errors[err] = dt_errors.get(err, 0) + 1
            
    if nl_c and not ld_c:
        nl_correct_ld_wrong.append(qid)
        err = p_ld.get('error')
        if err:
            ld_errors[err] = ld_errors.get(err, 0) + 1

print(f"\nQuestions where NL is correct but DT is wrong: {len(nl_correct_dt_wrong)}")
if dt_errors:
    print("Most common DT errors for these questions:")
    for e, c in sorted(dt_errors.items(), key=lambda x: -x[1])[:5]:
        print(f"  {c}x: {e}")
        
print(f"\nQuestions where NL is correct but LD is wrong: {len(nl_correct_ld_wrong)}")
if ld_errors:
    print("Most common LD errors for these questions:")
    for e, c in sorted(ld_errors.items(), key=lambda x: -x[1])[:5]:
        print(f"  {c}x: {e}")

# Also compare DT vs LD generally
dt_pipeline_errors = sum(1 for p in data['DT']['predictions'] if p.get('error'))
nl_pipeline_errors = sum(1 for p in data['NL']['predictions'] if p.get('error'))
ld_pipeline_errors = sum(1 for p in data['LD']['predictions'] if p.get('error'))

print(f"\nTotal Pipeline Errors: DT={dt_pipeline_errors}, NL={nl_pipeline_errors}, LD={ld_pipeline_errors}")

# Let's print out an example where NL was correct but DT/LD failed with an error, to see the error details
example_q = None
for qid in nl_correct_dt_wrong:
    dt_pred = preds['DT'].get(qid)
    if dt_pred and dt_pred.get('error'):
        example_q = qid
        break

if example_q:
    print(f"\nExample Question ID: {example_q}")
    print(f"NL Diagnostic: {preds['NL'].get(example_q, {}).get('routing_diagnostics')}")
    print(f"DT Error: {preds['DT'].get(example_q, {}).get('error')}")
    print(f"LD Error: {preds['LD'].get(example_q, {}).get('error')}")
    print(f"DT Diagnostic: {preds['DT'].get(example_q, {}).get('routing_diagnostics')}")
    print(f"LD Diagnostic: {preds['LD'].get(example_q, {}).get('routing_diagnostics')}")
