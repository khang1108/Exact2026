import json
import re
import asyncio
import httpx
from pathlib import Path

Z3_URL = "https://api.iamphuckhang.dev/z3"
DATA_PATH = Path("src/exact/datasets/exact/Logic_Based_Educational_Queries.json")

def parse_mcq_options(question_text: str) -> tuple[str, dict]:
    lines = question_text.split("\n")
    stem_lines, options = [], {}
    for line in lines:
        m = re.match(r"^([A-D])\.\s*(.*)", line.strip())
        if m:
            options[m.group(1)] = m.group(2).strip()
        elif not options:
            stem_lines.append(line)
    return "\n".join(stem_lines).strip(), options

def flatten_instances(groups: list) -> list[dict]:
    instances = []
    for g_idx, group in enumerate(groups):
        for q_idx, (question, gold) in enumerate(zip(group["questions"], group["answers"])):
            stem, options = parse_mcq_options(question)
            instances.append({
                "id": f"logic_{g_idx:04d}_{q_idx:02d}",
                "premises": group["premises-NL"],
                "question": stem,
                "options": options or None,
                "gold": gold,
                "q_type": "mcq" if options else "ynu",
            })
    return instances

async def call_z3(client, inst):
    payload = {
        "id": inst["id"],
        "query": inst["question"],
        "premises": inst["premises"],
    }
    if inst["options"]:
        payload["options"] = inst["options"]

    try:
        r = await client.post(Z3_URL, json=payload, timeout=60.0)
        if r.status_code == 200:
            body = r.json()
            return {
                "id": inst["id"],
                "q_type": inst["q_type"],
                "gold": inst["gold"],
                "pred": body.get("answer"),
                "fol": body.get("fol"),
                "error": None
            }
        else:
            return {
                "id": inst["id"],
                "q_type": inst["q_type"],
                "gold": inst["gold"],
                "pred": None,
                "fol": None,
                "error": f"HTTP {r.status_code}: {r.text}"
            }
    except Exception as e:
        return {
            "id": inst["id"],
            "q_type": inst["q_type"],
            "gold": inst["gold"],
            "pred": None,
            "fol": None,
            "error": str(e)
        }

async def main():
    with open(DATA_PATH, "r") as f:
        raw = json.load(f)
    
    instances = flatten_instances(raw)
    subset = instances[:50]
    
    print(f"Running Z3 pipeline on first {len(subset)} samples...")
    results = []
    
    # Run requests concurrently with a limit
    sem = asyncio.Semaphore(5)
    async with httpx.AsyncClient() as client:
        async def worker(inst):
            async with sem:
                res = await call_z3(client, inst)
                print(f"Done {inst['id']}: pred={res['pred']}, gold={res['gold']}, correct={res['pred'] == res['gold']}")
                return res
        
        results = await asyncio.gather(*(worker(i) for i in subset))
    
    # Write report
    report_lines = [
        "# FOL Representation and Predictions for the First 50 Samples\n",
        "| ID | Type | Gold | Pred | Correct | Error |",
        "|---|---|---|---|---|---|",
    ]
    
    for r in results:
        correct = "✓" if r["pred"] == r["gold"] else "✗"
        report_lines.append(f"| {r['id']} | {r['q_type']} | {r['gold']} | {r['pred']} | {correct} | {r['error'] or ''} |")
    
    report_lines.append("\n## Detailed FOL Dumps\n")
    
    for r, inst in zip(results, subset):
        report_lines.append(f"### {r['id']} (Gold: {r['gold']}, Pred: {r['pred']})")
        report_lines.append("**Premises (NL):**")
        for i, p in enumerate(inst["premises"], 1):
            report_lines.append(f"{i}. {p}")
        report_lines.append("\n**Question / Options (NL):**")
        report_lines.append(f"- Question: {inst['question']}")
        if inst["options"]:
            for opt_k, opt_v in inst["options"].items():
                report_lines.append(f"  - {opt_k}: {opt_v}")
        
        report_lines.append("\n**Parsed FOL:**")
        if r["fol"]:
            report_lines.append("```text")
            report_lines.append(r["fol"].strip())
            report_lines.append("```")
        else:
            report_lines.append("*No FOL available*")
        report_lines.append("\n---\n")
        
    out_path = Path("outputs/fol_dump_first_50_after.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"Report written to {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
