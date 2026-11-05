#!/usr/bin/env python3
"""
Benchmark the 19 Previously Failed LiveCodeBench Problems Across 3 Base Models:
1. ApexCode-14B (Qwen 2.5 Coder 14B)
2. ApexPhi-14B (Microsoft Phi-4 14.7B)
3. ApexMistral-22B (Mistral Codestral 22B)

Evaluates on real LiveCodeBench test suites to measure exact recovery rates,
algorithmic correctness, throughput (tok/s), and AST syntax validity.

Author: Nilesh Chakraborty / AGY Swarm Systems
"""

import os
import sys
import time
import json
import ast
import argparse
import urllib.request
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVECODEBENCH_DIR = "/Users/nileshchakraborty/workspace/LiveCodeBench"
APEX_DIR = "/Users/nileshchakraborty/workspace/apex-code-36gb"

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LIVECODEBENCH_DIR)
sys.path.insert(0, APEX_DIR)

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from scripts.ast_linter_tool import KarpathyASTLinter, audit_code

OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434")

# The 19 problems that failed in the standard zero-shot 50-problem evaluation
FAILED_50_IDS = [
    "1873_B", "1873_D", "1883_B", "1883_C", "1899_A", "1899_C",
    "2754", "2777", "2784", "2808", "2817", "2827",
    "2832", "2833", "2837", "2839", "2849", "2850", "2854"
]

COMPETITIVE_SCAFFOLDING = """
CRITICAL INSTRUCTIONS FOR EXECUTION:
1. Wrap the entire solution inside ```python ... ``` fences.
2. Read all input from standard input using fast I/O:
   import sys
   input_data = sys.stdin.read().split()
3. Print final results directly to standard output.
4. Do not hardcode sample outputs or make ungrounded assumptions.
"""


def extract_code_block(text: str) -> str:
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def query_model(model_name: str, prompt: str, starter_code: str = "") -> Dict[str, Any]:
    url = f"{OLLAMA_API.rstrip('/')}/api/generate"
    
    full_prompt = f"PROBLEM SPECIFICATION:\n{prompt}\n"
    if starter_code and starter_code.strip():
        full_prompt += f"\nSTARTER CODE TEMPLATE:\n```python\n{starter_code}\n```\n"
    else:
        full_prompt += f"\n{COMPETITIVE_SCAFFOLDING}\n"
        
    full_prompt += "\nOutput ONLY the complete, correct Python 3 code inside a ```python ``` block."

    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1024,
            "num_ctx": 16384
        }
    }
    
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()
    
    total_time = round(t1 - t0, 3)
    response_text = data.get("response", "")
    eval_count = data.get("eval_count", len(response_text.split()))
    eval_dur = data.get("eval_duration", 0) / 1e9
    speed = round(eval_count / eval_dur, 1) if eval_dur > 0 else round(eval_count / total_time, 1)
    
    code = extract_code_block(response_text)
    
    linter = KarpathyASTLinter()
    lint_res = linter.lint_python_code(code)
    try:
        ast.parse(code)
        ast_valid = True
    except SyntaxError:
        ast_valid = False
        
    return {
        "model": model_name,
        "latency_sec": total_time,
        "tokens": eval_count,
        "tok_per_sec": speed,
        "ast_valid": ast_valid,
        "linter_clean": lint_res["valid"],
        "code": code
    }


def main():
    parser = argparse.ArgumentParser(description="Three-Way Benchmark on Failed 19-Problem Set")
    parser.add_argument("--count", type=int, default=19, help="Number of failed problems to run (default: 19)")
    parser.add_argument("--offset", type=int, default=0, help="Offset index into failed set (default: 0)")
    parser.add_argument("--output", default="tests/lcb_failed_19_three_way_benchmark.json", help="Output path")
    args = parser.parse_args()

    models = ["apex-coder-14b", "apex-phi-14b", "apex-mistral-22b"]

    print("=" * 82)
    print("🥊 LIVECODEBENCH 19-PROBLEM FAILURE REMEDIATION SUITE")
    print(f"Target Problem Count: {args.count} | Offset: {args.offset}")
    print("Models Evaluated:", ", ".join(models))
    print("=" * 82)

    dataset = load_code_generation_dataset()
    target_all = [p for p in dataset if p.question_id in FAILED_50_IDS]
    target_all.sort(key=lambda p: FAILED_50_IDS.index(p.question_id))
    
    target_slice = target_all[args.offset : args.offset + args.count]
    print(f"Loaded {len(target_slice)} failed problems to evaluate.\n")

    # Storage per model
    samples_by_model = {m: [] for m in models}
    generations_by_model = {m: [] for m in models}
    metadata_by_model = {m: [] for m in models}

    for idx, prob in enumerate(target_slice, 1):
        diff = prob.difficulty.value
        print(f"[{idx}/{len(target_slice)}] {prob.question_id}: {prob.question_title[:32]} ({diff})")
        print("-" * 82)

        for model in models:
            res = query_model(model, prob.question_content, prob.starter_code or "")
            print(f"  ⚡ {model:<18}: {res['tokens']:>3} tok | {res['latency_sec']:>5.2f}s | {res['tok_per_sec']:>4.1f} tok/s | AST: {res['ast_valid']}")
            
            samples_by_model[model].append(prob.get_evaluation_sample())
            generations_by_model[model].append([res["code"]])
            metadata_by_model[model].append({
                "question_id": prob.question_id,
                "question_title": prob.question_title,
                "difficulty": diff,
                "latency_sec": res["latency_sec"],
                "tokens": res["tokens"],
                "tok_per_sec": res["tok_per_sec"],
                "ast_valid": res["ast_valid"],
                "code": res["code"]
            })
        print()

    # Run official test execution sandbox for all 3 models
    print("=" * 82)
    print("▶ Running Official LiveCodeBench Execution Test Suites for All 3 Models...")
    print("=" * 82)

    scorecard = {}

    for model in models:
        print(f"\nEvaluating test suites for {model}...")
        t0 = time.perf_counter()
        metrics, graded, _ = codegen_metrics(
            samples_by_model[model],
            generations_by_model[model],
            k_list=[1],
            num_process_evaluate=4,
            timeout=6
        )
        eval_dur = round(time.perf_counter() - t0, 2)
        pass_at_1 = float(metrics.get("pass@1", 0.0))
        
        passed_problems = []
        for p_idx, prob in enumerate(target_slice):
            res_list = graded.get(p_idx, [])
            all_passed = bool(res_list and all(x > 0 for x in res_list[0]) and len(res_list[0]) > 0) if res_list else False
            passed_cases = sum(1 for x in res_list[0] if x > 0) if res_list else 0
            total_cases = len(res_list[0]) if res_list else 0
            
            metadata_by_model[model][p_idx]["passed"] = all_passed
            metadata_by_model[model][p_idx]["passed_cases"] = passed_cases
            metadata_by_model[model][p_idx]["total_cases"] = total_cases
            
            if all_passed:
                passed_problems.append(prob.question_id)

        avg_speed = round(sum(m["tok_per_sec"] for m in metadata_by_model[model]) / len(metadata_by_model[model]), 1)
        total_time = round(sum(m["latency_sec"] for m in metadata_by_model[model]), 2)
        ast_rate = round(sum(1 for m in metadata_by_model[model] if m["ast_valid"]) / len(metadata_by_model[model]) * 100, 1)

        scorecard[model] = {
            "pass_at_1_pct": round(pass_at_1 * 100, 1),
            "passed_count": len(passed_problems),
            "total_count": len(target_slice),
            "passed_problems": passed_problems,
            "avg_tok_per_sec": avg_speed,
            "total_latency_sec": total_time,
            "ast_valid_pct": ast_rate,
            "eval_duration_sec": eval_dur
        }

    # Print Final Comparative Matrix
    print("\n" + "=" * 82)
    print("🏆 FINAL COMPARATIVE SCORECARD ON PREVIOUSLY FAILED PROBLEMS")
    print("=" * 82)
    header = f"{'Metric':<32} | " + " | ".join(f"{m:<18}" for m in models)
    print(header)
    print("-" * 82)
    
    print(f"{'Pass@1 Recovery Rate':<32} | " + " | ".join(f"{scorecard[m]['passed_count']}/{scorecard[m]['total_count']} ({scorecard[m]['pass_at_1_pct']}%)".ljust(18) for m in models))
    print(f"{'Average Speed (tok/s)':<32} | " + " | ".join(f"{scorecard[m]['avg_tok_per_sec']} tok/s".ljust(18) for m in models))
    print(f"{'Total Latency':<32} | " + " | ".join(f"{scorecard[m]['total_latency_sec']}s".ljust(18) for m in models))
    print(f"{'AST Syntax Validity':<32} | " + " | ".join(f"{scorecard[m]['ast_valid_pct']}%".ljust(18) for m in models))
    print("=" * 82)

    print("\n📋 PROBLEM RECOVERY BY MODEL:")
    for prob in target_slice:
        pid = prob.question_id
        statuses = []
        for m in models:
            p_data = next((item for item in metadata_by_model[m] if item["question_id"] == pid), None)
            is_pass = p_data.get("passed", False) if p_data else False
            statuses.append(f"{'✅' if is_pass else '❌'} {m[:10]}")
        print(f"  • {pid:<10}: {prob.question_title[:24]:<24} | " + " | ".join(statuses))

    # Persist report to disk
    out_file = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    report = {
        "suite": "LiveCodeBench Failed Problems Remediation",
        "models": models,
        "problem_count": len(target_slice),
        "scorecard": scorecard,
        "problems_metadata": metadata_by_model
    }
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Saved comprehensive report to: {out_file}\n")


if __name__ == "__main__":
    main()
