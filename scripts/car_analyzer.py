#!/usr/bin/env python3
"""
CAR Task (Cause & Analysis Request) Engine & Zero-Assumption Scope Verifier.

Implements:
1. Root-cause defect severity classification:
   - Type A (Surface Defect): Low blast radius (typos, imports, fast I/O, integer division)
     -> Triggers immediate single-shot deterministic AST repair (System 1).
   - Type B (Root-Cause Epistemic Defect): High blast radius (algorithmic complexity TLE,
     broken invariants, missing specification, unhandled constraints)
     -> Emits BLOCKED_MISSING_INFO or CAR_DEFECT ticket and escalates to System 2 RecursiveMAS.
2. Zero-Assumption & Strict Bounded Scope Invariant:
   - Verifies subagents never guess unspecified schemas or violate contracted input/output boundaries.

Author: Nilesh Chakraborty / AGY Swarm Systems
"""

import os
import sys
import re
import ast
import json
import time
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.epistemic_cache import EpistemicMemoryBus


# Common patterns indicative of Type A (Surface) vs Type B (Root-Cause) defects
TYPE_A_PATTERNS = [
    (r"NameError:\s+name\s+'(math|sys|collections|itertools|heapq|bisect)'\s+is\s+not\s+defined", "missing_standard_import"),
    (r"ModuleNotFoundError:\s+No module named", "missing_import"),
    (r"SyntaxError:\s+invalid syntax", "syntax_typo"),
    (r"IndentationError:", "indentation_defect"),
    (r"TypeError:\s+float object cannot be interpreted as an integer", "float_int_division"),
    (r"ZeroDivisionError:", "local_zero_division"),
    (r"AttributeError:\s+'list' object has no attribute 'add'", "api_typo_append"),
    (r"RecursionError:\s+maximum recursion depth exceeded", "recursion_limit_missing"),
]

TYPE_B_PATTERNS = [
    (r"Time Limit Exceeded|TLE|timed out|TimeoutExpired", "time_limit_exceeded_algorithmic_complexity"),
    (r"Memory Limit Exceeded|MLE|OutOfMemory", "memory_limit_exceeded_spatial_complexity"),
    (r"Wrong Answer|WA|AssertionError|Test output mismatch", "algorithmic_invariant_violation"),
    (r"BLOCKED_MISSING_INFO|Unspecified schema|Ambiguous input", "missing_specification"),
    (r"IndexError:\s+list index out of range", "boundary_invariant_failure")
]


class CARAnalyzer:
    """
    Cause & Analysis Request (CAR) Engine.
    Evaluates defects and enforces zero-assumption contracts.
    """
    def __init__(self, memory_bus: Optional[EpistemicMemoryBus] = None):
        self.bus = memory_bus or EpistemicMemoryBus()

    def analyze_defect(
        self,
        code: str,
        defect_message: str,
        problem_spec: str = "",
        subagent_id: str = "agent_executor"
    ) -> Dict[str, Any]:
        """
        Classify defect into Type A (Surface) or Type B (Root-Cause Epistemic).
        """
        start_t = time.time()
        classification = "TYPE_A"
        root_cause = "unknown_surface_defect"
        confidence = 0.80
        action = "SINGLE_SHOT_AST_REPAIR"
        blast_radius = "LOCAL_LINE"
        missing_entity = None
        remediation_prompt = None

        # Check for Type B indicators first (high blast radius)
        for pattern, cause_label in TYPE_B_PATTERNS:
            if re.search(pattern, defect_message, re.IGNORECASE):
                classification = "TYPE_B"
                root_cause = cause_label
                confidence = 0.95
                action = "ESCALATE_TO_RECURSIVEMAS"
                blast_radius = "SYSTEMIC_ALGORITHM"
                break

        # Check algorithmic complexity mismatch in problem_spec vs code
        if classification == "TYPE_A":
            complexity_flag = self._detect_complexity_mismatch(code, problem_spec)
            if complexity_flag:
                classification = "TYPE_B"
                root_cause = complexity_flag["cause"]
                confidence = 0.90
                action = "ESCALATE_TO_RECURSIVEMAS"
                blast_radius = "SYSTEMIC_ALGORITHM"
                missing_entity = complexity_flag.get("missing_optimal_structure")

        # If not Type B, check Type A surface patterns
        if classification == "TYPE_A":
            matched = False
            for pattern, cause_label in TYPE_A_PATTERNS:
                m = re.search(pattern, defect_message)
                if m:
                    root_cause = cause_label
                    action = "SINGLE_SHOT_AST_REPAIR"
                    blast_radius = "LOCAL_LINE"
                    matched = True
                    break
            if not matched:
                # Syntax / AST check
                try:
                    ast.parse(code)
                    # If syntax is valid but there's a test failure not caught by pattern, evaluate as Type B
                    classification = "TYPE_B"
                    root_cause = "logic_contract_mismatch"
                    action = "ESCALATE_TO_RECURSIVEMAS"
                    blast_radius = "LOGIC_INVARIANT"
                except SyntaxError as e:
                    root_cause = "syntax_error"
                    action = "SINGLE_SHOT_AST_REPAIR"
                    blast_radius = f"LINE_{e.lineno}"

        # If Type B, post an Epistemic Defect Ticket to the shared memory bus
        ticket_id = None
        if classification == "TYPE_B":
            ticket_id = self.bus.post_ticket(
                ticket_type="CAR_DEFECT_ROOT_CAUSE",
                subagent_id=subagent_id,
                missing_entity=missing_entity or root_cause,
                blocking_reason=f"Defect '{root_cause}' requires multi-stage epistemic re-planning: {defect_message[:150]}",
                payload={
                    "defect_message": defect_message,
                    "root_cause": root_cause,
                    "code_snippet": code[:500]
                }
            )

        elapsed = (time.time() - start_t) * 1000.0
        return {
            "classification": classification, # "TYPE_A" or "TYPE_B"
            "root_cause": root_cause,
            "blast_radius": blast_radius,
            "confidence": confidence,
            "recommended_action": action,
            "ticket_id": ticket_id,
            "elapsed_ms": round(elapsed, 2)
        }

    def verify_zero_assumption_invariant(
        self,
        code: str,
        declared_contract: Dict[str, Any],
        subagent_id: str = "subagent_worker"
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Enforce Strict Bounded Scope & Zero-Assumption Protocol:
        1. Subagent must NOT introduce unrequested global side-effects.
        2. Subagent must NOT invent ungrounded external dependencies.
        3. If required schemas are missing, emits BLOCKED_MISSING_INFO.
        Returns: (is_valid, violation_reason, ticket_id)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error in code: {e}", None

        # Check for disallowed imports
        allowed_imports = set(declared_contract.get("allowed_imports", ["sys", "math", "collections", "heapq", "itertools", "bisect"]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_root = alias.name.split(".")[0]
                    if mod_root not in allowed_imports:
                        reason = f"Subagent violated bounded scope: unauthorized import '{mod_root}'"
                        t_id = self.bus.post_ticket(
                            ticket_type="BLOCKED_MISSING_INFO",
                            subagent_id=subagent_id,
                            missing_entity=f"import_authorization_{mod_root}",
                            blocking_reason=reason
                        )
                        return False, reason, t_id

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod_root = node.module.split(".")[0]
                    if mod_root not in allowed_imports:
                        reason = f"Subagent violated bounded scope: unauthorized import from '{mod_root}'"
                        t_id = self.bus.post_ticket(
                            ticket_type="BLOCKED_MISSING_INFO",
                            subagent_id=subagent_id,
                            missing_entity=f"import_authorization_{mod_root}",
                            blocking_reason=reason
                        )
                        return False, reason, t_id

        # Check for guessed hardcoded mock constants (hallucinated magic data)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.upper() in ["MOCK_DATA", "ASSUMED_SCHEMA", "DEFAULT_PAYLOAD"]:
                        reason = f"Subagent violated zero-assumption invariant: fabricated '{target.id}'"
                        t_id = self.bus.post_ticket(
                            ticket_type="BLOCKED_MISSING_INFO",
                            subagent_id=subagent_id,
                            missing_entity=target.id,
                            blocking_reason=reason
                        )
                        return False, reason, t_id

        return True, None, None

    def _detect_complexity_mismatch(self, code: str, problem_spec: str) -> Optional[Dict[str, str]]:
        """
        Detects whether code has nested loops O(N^2) while problem spec indicates N >= 10^5 (guaranteed TLE).
        """
        n_match = re.search(r"N\s*\\le\s*2\s*\\cdot\s*10\^5|N\s*<=\s*200000|10\^5", problem_spec)
        if not n_match:
            return None

        # Check for nested loops in AST
        try:
            tree = ast.parse(code)
            nested_loop_found = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    for child in ast.walk(node):
                        if child is not node and isinstance(child, (ast.For, ast.While)):
                            nested_loop_found = True
                            break
                if nested_loop_found:
                    break

            if nested_loop_found:
                return {
                    "cause": "quadratic_complexity_on_large_N",
                    "missing_optimal_structure": "O(N_log_N)_or_O(N)_algorithm"
                }
        except Exception:
            pass

        return None


# -----------------------------------------------------------------------------
# Standalone CLI / Verification
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing CAR Task Analyzer & Zero-Assumption Invariant ===")
    bus = EpistemicMemoryBus()
    car = CARAnalyzer(bus)

    # 1. Surface Defect (Type A)
    type_a_code = "print(math.sqrt(16))"
    type_a_err = "NameError: name 'math' is not defined"
    res_a = car.analyze_defect(type_a_code, type_a_err, problem_spec="")
    print(f"Surface Defect Result: {res_a['classification']} | Cause: {res_a['root_cause']} | Action: {res_a['recommended_action']}")
    assert res_a["classification"] == "TYPE_A"
    assert res_a["recommended_action"] == "SINGLE_SHOT_AST_REPAIR"

    # 2. Epistemic Defect (Type B - TLE / Algorithmic Mismatch)
    type_b_code = """
def solve():
    for i in range(len(arr)):
        for j in range(len(arr)):
            pass
"""
    type_b_spec = "Constraint: N <= 200000. Time limit 1.0s."
    res_b = car.analyze_defect(type_b_code, "Time Limit Exceeded on test 14", problem_spec=type_b_spec)
    print(f"Epistemic Defect Result: {res_b['classification']} | Cause: {res_b['root_cause']} | Action: {res_b['recommended_action']} | Ticket: {res_b['ticket_id']}")
    assert res_b["classification"] == "TYPE_B"
    assert res_b["recommended_action"] == "ESCALATE_TO_RECURSIVEMAS"
    assert res_b["ticket_id"] is not None

    # 3. Zero-Assumption Scope Verification
    violating_code = """
import requests
def process():
    MOCK_DATA = {'id': 123}
    return MOCK_DATA
"""
    contract = {"allowed_imports": ["sys", "math"]}
    valid, reason, t_id = car.verify_zero_assumption_invariant(violating_code, contract, subagent_id="subagent_test")
    print(f"Zero-Assumption Check: Valid={valid} | Reason={reason} | Ticket={t_id}")
    assert valid is False
    assert "unauthorized import 'requests'" in reason

    print("\nTelemetry:", bus.get_telemetry())
    print("\n[SUCCESS] CAR Analyzer passed all verification checks!")
