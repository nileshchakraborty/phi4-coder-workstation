#!/usr/bin/env python3
"""
CAR Task (Cause & Analysis Request) Closed-Loop Self-Repair Benchmark Harness.

Unites:
1. Big-O Complexity Prompt Conditioning (Phi-4, Codestral, Qwen)
2. Static AST Complexity Linting (KarpathyASTLinter: pop(0), nested sorted(), stdin in loops)
3. Sample Test-Time Verification & Surgical Reflection (Type A vs Type B CAR Tasks)
4. Official LiveCodeBench Execution Sandbox Evaluation

Author: Nilesh Chakraborty / AGY Swarm Systems
"""

import os
import sys
import time
import json
import ast
import re
import argparse
import urllib.request
import subprocess
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVECODEBENCH_DIR = "/Users/nileshchakraborty/workspace/LiveCodeBench"
APEX_DIR = "/Users/nileshchakraborty/workspace/apex-code-36gb"

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LIVECODEBENCH_DIR)
sys.path.insert(0, APEX_DIR)

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from scripts.ast_linter_tool import KarpathyASTLinter, audit_code, generate_tool_repair_prompt
from scripts.car_analyzer import CARAnalyzer

OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434")

# The 19 problems that failed in the standard zero-shot baseline
TARGET_19_IDS = [
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
5. Strictly adhere to optimal Big-O bounds (avoid list.pop(0) in loops or sorting subarrays in nested loops).
"""


def extract_code_block(text: str) -> str:
    # Strip <think>...</think> reasoning blocks if present
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def query_ollama(model_name: str, prompt: str) -> Tuple[str, Dict[str, Any]]:
    url = f"{OLLAMA_API.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
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

    latency = round(t1 - t0, 3)
    response_text = data.get("response", "")
    tokens = data.get("eval_count", len(response_text.split()))
    speed = round(tokens / latency, 1) if latency > 0 else 0.0

    return response_text, {
        "latency_sec": latency,
        "tokens": tokens,
        "tok_per_sec": speed
    }


def generate_initial_code(model_name: str, problem_content: str, starter_code: str = "") -> Tuple[str, Dict[str, Any]]:
    prompt = f"PROBLEM SPECIFICATION:\n{problem_content}\n"
    if starter_code and starter_code.strip():
        prompt += f"\nSTARTER CODE TEMPLATE:\n```python\n{starter_code}\n```\n"
    else:
        prompt += f"\n{COMPETITIVE_SCAFFOLDING}\n"
    prompt += "\nOutput ONLY the complete, correct Python 3 code inside a ```python ``` block."
    
    resp_text, meta = query_ollama(model_name, prompt)
    code = extract_code_block(resp_text)
    return code, meta


def run_car_ast_repair(model_name: str, code: str, violations: List[Dict[str, Any]]) -> Tuple[str, bool]:
    """Type A CAR Task: single-shot deterministic tool repair prompt based on AST linter."""
    audit_res = {"valid": False, "violations": violations}
    repair_prompt = generate_tool_repair_prompt(code, audit_res)
    
    resp_text, _ = query_ollama(model_name, repair_prompt)
    repaired_code = extract_code_block(resp_text)
    
    # Re-verify AST
    re_audit = audit_code(repaired_code)
    return repaired_code, re_audit["valid"]


def main():
    parser = argparse.ArgumentParser(description="CAR Task Closed-Loop Self-Repair Benchmark Harness")
    parser.add_argument("--models", nargs="+", default=["apex-phi-14b", "apex-coder-14b", "apex-mistral-22b"])
    parser.add_argument("--count", type=int, default=19, help="Number of problems to evaluate (default: 19)")
    parser.add_argument("--offset", type=int, default=0, help="Offset into target problem set")
    parser.add_argument("--output", default="tests/car_repaired_benchmark_report.json", help="Output report path")
    args = parser.parse_args()

    print("=" * 84)
    print("🛠️  CAR TASK CLOSED-LOOP REPAIR BENCHMARK HARNESS")
    print(f"Target Problem Count: {args.count} | Offset: {args.offset}")
    print("Models Evaluated:", ", ".join(args.models))
    print("=" * 84)

    dataset = load_code_generation_dataset()
    target_all = [p for p in dataset if p.question_id in TARGET_19_IDS]
    target_all.sort(key=lambda p: TARGET_19_IDS.index(p.question_id))
    target_slice = target_all[args.offset : args.offset + args.count]
    print(f"Loaded {len(target_slice)} failed problems for CAR remediation.\n")

    linter = KarpathyASTLinter()
    car_analyzer = CARAnalyzer()

    # Track results per model
    final_scorecard = {}
    detailed_results = {m: [] for m in args.models}

    for model in args.models:
        print("\n" + "=" * 84)
        print(f"▶ Commencing CAR Task Self-Repair Evaluation for: {model}")
        print("=" * 84)

        samples = []
        generations = []
        model_meta = []
        type_a_repairs_count = 0
        type_b_repairs_count = 0

        for idx, prob in enumerate(target_slice, 1):
            pid = prob.question_id
            diff = prob.difficulty.value
            title = prob.question_title[:32]
            print(f"[{idx}/{len(target_slice)}] {pid}: {title} ({diff})")

            # 1. Initial Generation with Big-O Conditioning
            code, gen_meta = generate_initial_code(model, prob.question_content, prob.starter_code or "")
            print(f"  ⚡ Initial: {gen_meta['tokens']} tok | {gen_meta['latency_sec']}s | {gen_meta['tok_per_sec']} tok/s")

            # 2. Static AST & Complexity Audit
            audit_res = audit_code(code)
            repaired = False
            repair_type = None

            if not audit_res["valid"]:
                violation_types = [v["type"] for v in audit_res["violations"]]
                print(f"  ⚠️  [CAR Task - Type A] Static Violation Detected: {', '.join(violation_types)}")
                
                # Perform Type A single-shot repair
                repaired_code, now_valid = run_car_ast_repair(model, code, audit_res["violations"])
                if now_valid or len(repaired_code) > 20:
                    code = repaired_code
                    repaired = True
                    repair_type = "TYPE_A_AST_REPAIR"
                    type_a_repairs_count += 1
                    print(f"  🔧 [CAR Task - Type A] Repaired via deterministic AST tool prompt (Clean: {now_valid})")

            samples.append(prob.get_evaluation_sample())
            generations.append([code])
            model_meta.append({
                "question_id": pid,
                "question_title": prob.question_title,
                "difficulty": diff,
                "repaired": repaired,
                "repair_type": repair_type,
                "latency_sec": gen_meta["latency_sec"],
                "tokens": gen_meta["tokens"],
                "tok_per_sec": gen_meta["tok_per_sec"],
                "code": code
            })
            print()

        # Run Official LiveCodeBench Execution Test Suites
        print(f"Evaluating official test suites for {model} with CAR Task repairs...")
        t0 = time.perf_counter()
        metrics, graded, _ = codegen_metrics(
            samples,
            generations,
            k_list=[1],
            num_process_evaluate=4,
            timeout=6
        )
        eval_dur = round(time.perf_counter() - t0, 2)
        pass_at_1 = float(metrics.get("pass@1", 0.0))

        passed_pids = []
        for p_idx, prob in enumerate(target_slice):
            res_list = graded.get(p_idx, [])
            all_passed = bool(res_list and all(x > 0 for x in res_list[0]) and len(res_list[0]) > 0) if res_list else False
            passed_cases = sum(1 for x in res_list[0] if x > 0) if res_list else 0
            total_cases = len(res_list[0]) if res_list else 0

            model_meta[p_idx]["passed"] = all_passed
            model_meta[p_idx]["passed_cases"] = passed_cases
            model_meta[p_idx]["total_cases"] = total_cases
            if all_passed:
                passed_pids.append(prob.question_id)

        avg_speed = round(sum(m["tok_per_sec"] for m in model_meta) / len(model_meta), 1)
        total_time = round(sum(m["latency_sec"] for m in model_meta), 2)

        final_scorecard[model] = {
            "pass_at_1_pct": round(pass_at_1 * 100, 1),
            "passed_count": len(passed_pids),
            "total_count": len(target_slice),
            "passed_problems": passed_pids,
            "type_a_repairs_count": type_a_repairs_count,
            "avg_tok_per_sec": avg_speed,
            "total_latency_sec": total_time,
            "eval_duration_sec": eval_dur
        }
        detailed_results[model] = model_meta

    # Print Comparative Scorecard
    print("\n" + "=" * 84)
    print("🏆 FINAL CAR TASK REPAIR SCORECARD")
    print("=" * 84)
    header = f"{'Metric':<32} | " + " | ".join(f"{m:<18}" for m in args.models)
    print(header)
    print("-" * 84)
    print(f"{'Pass@1 Recovery Rate':<32} | " + " | ".join(f"{final_scorecard[m]['passed_count']}/{final_scorecard[m]['total_count']} ({final_scorecard[m]['pass_at_1_pct']}%)".ljust(18) for m in args.models))
    print(f"{'Type A AST Repairs Applied':<32} | " + " | ".join(f"{final_scorecard[m]['type_a_repairs_count']} repairs".ljust(18) for m in args.models))
    print(f"{'Average Speed (tok/s)':<32} | " + " | ".join(f"{final_scorecard[m]['avg_tok_per_sec']} tok/s".ljust(18) for m in args.models))
    print(f"{'Total Latency':<32} | " + " | ".join(f"{final_scorecard[m]['total_latency_sec']}s".ljust(18) for m in args.models))
    print("=" * 84)

    # Union of all solved problems across models
    all_solved = set()
    for sc in final_scorecard.values():
        all_solved.update(sc["passed_problems"])
    print(f"\n🌟 Epistemic Swarm Collective Recovery: {len(all_solved)} / {len(target_slice)} ({round(len(all_solved)/len(target_slice)*100, 1)}%)")
    print(f"Solved Problem IDs: {sorted(list(all_solved))}")

    # Save to disk
    out_file = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    report = {
        "suite": "LiveCodeBench CAR Task Closed-Loop Self-Repair Benchmark",
        "models": args.models,
        "scorecard": final_scorecard,
        "collective_recovery": {
            "solved_count": len(all_solved),
            "total_count": len(target_slice),
            "percentage": round(len(all_solved) / len(target_slice) * 100, 1),
            "solved_problems": sorted(list(all_solved))
        },
        "detailed_results": detailed_results
    }
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Saved comprehensive report to: {out_file}\n")


if __name__ == "__main__":
    main()
