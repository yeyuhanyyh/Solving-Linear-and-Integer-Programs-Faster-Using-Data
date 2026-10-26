"""Compare the sealed development and new confirmation evaluator statically.

This tool parses source only. It never imports either evaluator, opens data or
checkpoints, constructs models, generates costs, or executes a solver.

The comparison covers the complete direct-statement block beginning with
``begin = time.perf_counter()`` and ending immediately before ``row = dict``
in the per-cost loop. It also compares every row/summary keyword expression
except the explicitly listed phase and context identity fields. Source guards,
bank validation, initialization, and orchestration require separate review.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


OLD_FUNCTION = "evaluate_development"
NEW_FUNCTION = "evaluate_case_confirmation"
ROW_IDENTITY_FIELDS = frozenset({"guard", "experiment_phase", "confirmation"})
SUMMARY_IDENTITY_FIELDS = frozenset(
    {"phase", "no_claim_of_confirmation", "independent_confirmation"}
)


class EquivalenceStructureError(ValueError):
    """A required unique source structure was absent or ambiguous."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _assignment(node: ast.AST, name: str) -> ast.AST | None:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None
    target = node.targets[0]
    return node.value if isinstance(target, ast.Name) and target.id == name else None


def _dict_assignment(node: ast.AST, name: str) -> ast.Call | None:
    value = _assignment(node, name)
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "dict"
    ):
        return value
    return None


def _is_begin(node: ast.AST) -> bool:
    value = _assignment(node, "begin")
    return bool(
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "time"
        and value.func.attr == "perf_counter"
        and not value.args
        and not value.keywords
    )


def _one_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    found = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(found) != 1:
        raise EquivalenceStructureError(
            f"Expected one top-level {name!r} function; found {len(found)}"
        )
    return found[0]


def _cost_loop_parts(function: ast.FunctionDef) -> tuple[list[ast.stmt], ast.Call]:
    found: list[tuple[list[ast.stmt], ast.Call]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.For):
            continue
        starts = [i for i, item in enumerate(node.body) if _is_begin(item)]
        rows = [
            (i, value)
            for i, item in enumerate(node.body)
            if (value := _dict_assignment(item, "row")) is not None
        ]
        if not starts and not rows:
            continue
        if not starts or not rows:
            # A surrounding method loop may contain no direct per-cost block.
            # A loop containing only one sentinel is not the requested block.
            continue
        if len(starts) != 1 or len(rows) != 1:
            raise EquivalenceStructureError(
                "Ambiguous begin/row assignments in a potential per-cost loop"
            )
        begin_index = starts[0]
        row_index, row_call = rows[0]
        if begin_index >= row_index:
            raise EquivalenceStructureError("Per-cost row precedes its timer")
        found.append((node.body[begin_index:row_index], row_call))
    if len(found) != 1:
        raise EquivalenceStructureError(
            f"Expected one direct per-cost timer-to-row block; found {len(found)}"
        )
    return found[0]


def _summary(function: ast.FunctionDef) -> ast.Call:
    found = [
        value
        for node in ast.walk(function)
        if (value := _dict_assignment(node, "summary")) is not None
    ]
    if len(found) != 1:
        raise EquivalenceStructureError(
            f"Expected one summary=dict assignment; found {len(found)}"
        )
    return found[0]


def _keywords(call: ast.Call, allowed: frozenset[str]) -> tuple[dict[str, str], list[str]]:
    if call.args or any(item.arg is None for item in call.keywords):
        raise EquivalenceStructureError("Only explicit dict keyword expressions are permitted")
    result: dict[str, str] = {}
    ignored: list[str] = []
    seen: set[str] = set()
    for item in call.keywords:
        name = item.arg
        assert name is not None
        if name in seen:
            raise EquivalenceStructureError(f"Duplicate dict keyword {name!r}")
        seen.add(name)
        if name in allowed:
            ignored.append(name)
        else:
            result[name] = _dump(item.value)
    return result, sorted(ignored)


def _compare_keywords(old: ast.Call, new: ast.Call, allowed: frozenset[str]) -> dict:
    old_fields, old_ignored = _keywords(old, allowed)
    new_fields, new_ignored = _keywords(new, allowed)
    common = old_fields.keys() & new_fields.keys()
    return {
        "equivalent": old_fields == new_fields,
        "compared_keyword_count": len(old_fields),
        "old_keywords": sorted(old_fields),
        "new_keywords": sorted(new_fields),
        "missing_keywords": sorted(old_fields.keys() - new_fields.keys()),
        "extra_keywords": sorted(new_fields.keys() - old_fields.keys()),
        "changed_keyword_expressions": sorted(
            name for name in common if old_fields[name] != new_fields[name]
        ),
        "allowed_identity_fields": sorted(allowed),
        "ignored_old_identity_fields": old_ignored,
        "ignored_new_identity_fields": new_ignored,
    }


def verify(old_path: str | Path, new_path: str | Path) -> dict:
    """Return a deterministic report; mismatches fail with ``passed=False``.

    Missing or ambiguous syntax also fails closed in the report, while ordinary
    filesystem read errors remain exceptions. Neither file is imported.
    """
    old_path = Path(old_path).resolve()
    new_path = Path(new_path).resolve()
    old_raw = old_path.read_bytes()
    new_raw = new_path.read_bytes()
    report = {
        "schema": "generic-binary-confirmation-evaluator-static-equivalence-v1",
        "old_source": str(old_path),
        "new_source": str(new_path),
        "old_sha256": _sha(old_raw),
        "new_sha256": _sha(new_raw),
        "helper_sha256": _sha(Path(__file__).read_bytes()),
        "old_function": OLD_FUNCTION,
        "new_function": NEW_FUNCTION,
        "execution": "AST parsing only; evaluators not imported or executed",
        "scope": (
            "Per-cost timer-to-row scientific block and row/summary keyword "
            "expressions; identity fields excluded only as explicitly listed"
        ),
        "separate_review_required": [
            "global freeze and source/artifact guards",
            "bank generation, reference certification and validation",
            "model initialization and runtime binding before the per-cost block",
            "resume verification, output separation and orchestration",
        ],
    }
    try:
        old_function = _one_function(ast.parse(old_raw, filename=str(old_path)), OLD_FUNCTION)
        new_function = _one_function(ast.parse(new_raw, filename=str(new_path)), NEW_FUNCTION)
        old_block, old_row = _cost_loop_parts(old_function)
        new_block, new_row = _cost_loop_parts(new_function)
        old_dump = _dump(ast.Module(body=old_block, type_ignores=[]))
        new_dump = _dump(ast.Module(body=new_block, type_ignores=[]))
        report["scientific_block"] = {
            "equivalent": old_dump == new_dump,
            "old_statement_count": len(old_block),
            "new_statement_count": len(new_block),
            "old_ast_sha256": _sha(old_dump.encode()),
            "new_ast_sha256": _sha(new_dump.encode()),
            "first_line_old": old_block[0].lineno,
            "first_line_new": new_block[0].lineno,
            "last_line_old": old_block[-1].end_lineno,
            "last_line_new": new_block[-1].end_lineno,
        }
        report["row_keywords"] = _compare_keywords(old_row, new_row, ROW_IDENTITY_FIELDS)
        report["summary_keywords"] = _compare_keywords(
            _summary(old_function), _summary(new_function), SUMMARY_IDENTITY_FIELDS
        )
        report["passed"] = all(
            report[key]["equivalent"]
            for key in ("scientific_block", "row_keywords", "summary_keywords")
        )
    except (SyntaxError, EquivalenceStructureError) as error:
        report["passed"] = False
        report["structure_error"] = f"{type(error).__name__}: {error}"
    report["status"] = "PASS_STATIC_EQUIVALENCE" if report["passed"] else "FAIL_STATIC_EQUIVALENCE"
    return report


def _write_report_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"Refusing to overwrite a different report: {path}")
        return
    with path.open("xb") as stream:
        stream.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.old, args.new)
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if args.output is not None:
        _write_report_new(args.output, raw)
    print(raw.decode(), end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
