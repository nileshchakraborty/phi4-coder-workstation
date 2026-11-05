#!/usr/bin/env python3
"""
Phi-4 Coder Interactive CLI & Benchmark Runner.
Measures latency, TTFT, token throughput, and AST syntax correctness.
"""

import os
import sys
import time
import json
import ast
import argparse
import urllib.request
from typing import Dict, Any

MODEL_NAME = os.getenv("PHI4_MODEL", "phi4")
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434")


def query_phi4(prompt: str, stream: bool = False, temperature: float = 0.0) -> Dict[str, Any]:
    url = f"{OLLAMA_API.rstrip('/')}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
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
    
    # Extract code
    code = response_text
    if "```python" in response_text:
        code = response_text.split("```python")[1].split("```")[0].strip()
    elif "```" in response_text:
        code = response_text.split("```")[1].split("```")[0].strip()
        
    try:
        ast.parse(code)
        ast_valid = True
    except SyntaxError:
        ast_valid = False
        
    return {
        "model": MODEL_NAME,
        "latency_sec": total_time,
        "tokens": eval_count,
        "tok_per_sec": speed,
        "ast_valid": ast_valid,
        "code": code,
        "raw_response": response_text
    }


def main():
    parser = argparse.ArgumentParser(description="Phi-4 Coder CLI")
    parser.add_argument("--prompt", default="Implement an Augmented Interval Tree with point overlap search in Python 3. Output only typed code.", help="Prompt to run")
    args = parser.parse_args()
    
    print(f"🚀 Running Phi-4 Coder ({MODEL_NAME})...")
    print(f"Prompt: {args.prompt}\n")
    res = query_phi4(args.prompt)
    print("=" * 72)
    print(f"⚡ Latency: {res['latency_sec']}s | Tokens: {res['tokens']} | Speed: {res['tok_per_sec']} tok/s | AST Valid: {res['ast_valid']}")
    print("=" * 72)
    print(res["code"])


if __name__ == "__main__":
    main()
