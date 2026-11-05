#!/usr/bin/env python3
"""
ApexCode AST Linter Tool (Karpathy Software 3.0 External Tool Verification)
Performs deterministic static analysis on generated Python code to detect
common competitive programming regressions before sandbox execution.
"""

import ast
import re
from typing import List, Dict, Any, Optional

class CPCodeAuditVisitor(ast.NodeVisitor):
    def __init__(self, code: str):
        self.code = code
        self.issues: List[Dict[str, Any]] = []
        self.in_loop = 0
        self.read_called_outside = False
        self.read_called_inside_loop = False

    def visit_For(self, node: ast.For):
        self.in_loop += 1
        self.generic_visit(node)
        self.in_loop -= 1

    def visit_While(self, node: ast.While):
        self.in_loop += 1
        self.generic_visit(node)
        self.in_loop -= 1

    def visit_Call(self, node: ast.Call):
        # Check for sys.stdin.read() inside loops
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "stdin":
                if func_name == "read":
                    if self.in_loop > 0:
                        self.read_called_inside_loop = True
                    else:
                        self.read_called_outside = True
            
            # Check for list.pop(0) in loops (O(N) per pop -> O(N^2) TLE)
            if func_name == "pop" and self.in_loop > 0:
                if len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0:
                    self.issues.append({
                        "type": "LIST_POP_ZERO_IN_LOOP_TLE",
                        "line": getattr(node, "lineno", 0),
                        "message": "Calling `.pop(0)` on a standard Python list inside a loop is O(N) per operation, causing O(N^2) TLE on large inputs. Use `collections.deque` with `.popleft()` or maintain a two-pointer sliding window index."
                    })
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            # Check for sorted() inside nested loops (O(N^3) TLE)
            if func_name == "sorted" and self.in_loop >= 2:
                self.issues.append({
                    "type": "SORTED_IN_NESTED_LOOP_TLE",
                    "line": getattr(node, "lineno", 0),
                    "message": "Calling `sorted(...)` inside nested loops is O(N^3 log N) and will trigger Time Limit Exceeded (TLE) on N >= 1000. Maintain running state using two pointers, a frequency map, or prefix trees."
                })

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        # Flag float division `/` where operands are clearly integer loops or subscripts
        if isinstance(node.op, ast.Div):
            # Check if used inside subscript or range
            self.issues.append({
                "type": "FLOAT_DIVISION_WARNING",
                "line": getattr(node, "lineno", 0),
                "message": "Use integer division `//` instead of float division `/` to prevent precision errors and float types in array indexing."
            })
        self.generic_visit(node)

def audit_code(code: str) -> Dict[str, Any]:
    """Audits code for common competitive programming bugs."""
    violations = []
    
    # 1. Syntax Verification
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "valid": False,
            "violations": [{"type": "SYNTAX_ERROR", "line": e.lineno, "message": str(e)}],
            "fixed_code": None
        }

    # 2. AST Visitor checks
    visitor = CPCodeAuditVisitor(code)
    visitor.visit(tree)
    violations.extend(visitor.issues)

    if visitor.read_called_inside_loop:
        violations.append({
            "type": "REPEATED_STDIN_READ_IN_LOOP",
            "message": "Calling `sys.stdin.read()` inside a loop causes subsequent iterations to read EOF. Read tokens once with `data = sys.stdin.read().split()` and iterate with an index pointer."
        })

    # 3. Check for hardcoded indexing chain like a[0] * a[1] * a[2]...
    if re.search(r"\w+\[0\]\s*\*\s*\w+\[1\]\s*\*\s*\w+\[2\]", code):
        violations.append({
            "type": "HARDCODED_ARRAY_LENGTH_PRODUCT",
            "message": "Do not hardcode index multiplications like `a[0] * a[1] * a[2]...`. The array length `n` varies across test cases. Compute products dynamically using a loop or `math.prod(a)`."
        })

    # 4. Fast I/O check: Large inputs with input() in while/for
    if "sys.stdin" not in code and ("input()" in code):
        lines = code.splitlines()
        for idx, line in enumerate(lines, 1):
            if "for " in line and "in range(" in line and "input()" in line:
                violations.append({
                    "type": "SLOW_IO_WARNING",
                    "line": idx,
                    "message": "Use `sys.stdin.read().split()` for fast bulk I/O to avoid TLE on large competitive programming inputs."
                })
                break

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "fixed_code": None
    }

def generate_tool_repair_prompt(code: str, audit_result: Dict[str, Any]) -> str:
    """Formats deterministic diagnostic feedback for a single-pass model repair."""
    diagnostics = []
    for v in audit_result.get("violations", []):
        msg = v.get("message", "")
        v_type = v.get("type", "")
        diagnostics.append(f"- [{v_type}]: {msg}")
    
    diagnostic_text = "\n".join(diagnostics)
    
    repair_prompt = f"""The following solution has deterministic logic/syntax defects identified by our static analysis tool:

```python
{code}
```

Static Analysis Violations:
{diagnostic_text}

Please fix these exact violations and provide the corrected complete Python 3 solution inside ```python ```."""
    return repair_prompt

class KarpathyASTLinter:
    """Object-oriented wrapper for Karpathy Software 3.0 deterministic AST auditing and repairs."""
    def __init__(self):
        pass

    def lint_python_code(self, code: str) -> Dict[str, Any]:
        res = audit_code(code)
        issues = [v.get("message", "") for v in res.get("violations", [])]
        return {
            "valid": res["valid"],
            "issues": issues,
            "violations": res.get("violations", [])
        }

    def attempt_repair(self, code: str, issues: List[str]) -> str:
        # Deterministic line-level transformations for common Type A surface defects
        repaired = code
        # 1. Missing math import
        if any("math" in iss for iss in issues) and "import math" not in repaired:
            repaired = "import math\n" + repaired
        # 2. Missing sys import
        if any("sys" in iss for iss in issues) and "import sys" not in repaired:
            repaired = "import sys\n" + repaired
        # 3. Float division in indexing
        repaired = re.sub(r"\[(\w+)\s*/\s*(\w+)\]", r"[\1 // \2]", repaired)
        return repaired


if __name__ == "__main__":
    test_buggy_code = """import sys

def solve():
    data = sys.stdin.read().split()
    for _ in range(10):
        val = sys.stdin.read()
        x = val[0] * val[1] * val[2]
"""
    result = audit_code(test_buggy_code)
    print("Audit Result:")
    import json
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        print("\nRepair Prompt:")
        print(generate_tool_repair_prompt(test_buggy_code, result))
