"""Locate preserved runtime disagreements without executing a model or altering the baseline."""
import argparse
import json
from pathlib import Path
from protocol import digest

def collect(root):
    names=("common-bf16.json","native-control.json","common-candidate-a.json","native-candidate-a.json")
    arms=[json.loads((root/n).read_text()) for n in names]
    evaluation=json.loads((root/"evaluation.json").read_text())
    identity=digest(root/"evaluation.json")
    if any(a["evaluation_sha256"]!=identity for a in arms):
        raise ValueError("mismatched evaluation")
    rows=[{r["id"]:r for r in a["samples"]} for a in arms]
    inputs={r["id"]:r for r in evaluation["samples"]}
    if any(set(r)!=set(inputs) for r in rows):
        raise ValueError("missing runtime samples")
    cases=[]
    for sid,inp in inputs.items():
        expected=len(inp["input_ids"])-inp["score_start"]
        if any(len(r[sid]["score_lp"])!=expected or len(r[sid]["score_top1"])!=expected for r in rows):
            raise ValueError("mismatched target positions")
        for i in range(expected):
            values=[r[sid]["score_lp"][i] for r in rows]
            if abs(values[2]-values[3])>1:
                position=inp["score_start"]+i
                cases.append({"sample_id":sid,"target_index":i,"input_position":position,
                              "target_id":inp["input_ids"][position],
                              "logprob":dict(zip(names,values)),
                              "top1":dict(zip(names,[r[sid]["score_top1"][i] for r in rows])),
                              "prefix_ids":inp["input_ids"][:position],
                              "native_prefill_tokens":inp["score_start"],
                              "native_incremental_tokens":i})
    return {"evaluation_sha256":identity,"source_sha256":{n:digest(root/n) for n in names},
            "selection":"all absolute Candidate A native/common target log-probability differences above 1 nat",
            "cases":cases}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--record",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();report=collect(args.record);args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"outliers":len(report["cases"]),"evaluation_sha256":report["evaluation_sha256"]}))
