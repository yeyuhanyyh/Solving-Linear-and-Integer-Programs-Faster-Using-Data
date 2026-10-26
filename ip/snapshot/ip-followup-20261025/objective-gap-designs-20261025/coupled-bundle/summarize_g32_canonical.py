"""Summarize existing independent G32 development audits; no model or data imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLLOWUP = ROOT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reports = {n: read(ROOT / f"uniform-g32-{n}-canonical-completion-audit.json") for n in ("n69", "n36")}
    errors, pooled = [], {}
    for n, report in reports.items():
        assert report["status"] == "COMPLETE_AUDITED" and report["record_count"] == 288
        for method in ("ours", "neural_diving"):
            rows = [r for r in report["aggregates"] if r["method"] == method]
            assert len(rows) == 3 and sum(r["count"] for r in rows) == 144
            den = sum(r["ratio_denominator"] for r in rows)
            def weighted(key):
                return sum(r[key] * r["ratio_denominator"] for r in rows) / den
            pooled[n + "/" + method] = dict(count=144, ratio_N=den,
                recovered=sum(r["recovery_count"] for r in rows),
                recovery_percent=100 * sum(r["recovery_count"] for r in rows) / 144,
                original_profit_ratio=weighted("profit_ratio_mean_valid"),
                canonical_diagnostic_ratio=weighted("canonical_ratio_mean_same_primary_subset"),
                upgrade_selection_ratio_loss=weighted("upgrade_selection_ratio_loss_mean_same_primary_subset"),
                basic_completion_ratio_loss=weighted("basic_completion_ratio_loss_mean_same_primary_subset"),
                invalid_count=sum(r["invalid_primary_count"] for r in rows),
                late_count=sum(r["late_count"] for r in rows),
                canonical_violates_fixed_count=sum(r["canonical_violates_fixed_count_same_primary_subset"] for r in rows),
                canonical_diagnostic_never_replaces_primary=True)
    receipt = FOLLOWUP / "objective-gap-followup-20261025/uniform-gross-g32-n36-dev-bank-attachments-independent.json"
    receipt_data = read(receipt)
    assert receipt_data["manifest_sha256"] == reports["n36"]["manifest_sha256"]
    bound = reports["n36"]["bank_binding_evidence"]["cases"]
    for entry in receipt_data["entries"]:
        assert entry["byte_identical"] is True and bound[entry["case_id"]] == entry["sha256"]
    source_receipt = Path(receipt_data["source_receipt"])
    assert sha(source_receipt) == receipt_data["source_receipt_sha256"]
    run_seal_path = ROOT.parent / "adapter/generic-binary-v2/g32-short-prefix-n36-run-v1/run-seal.json"
    run_seal = read(run_seal_path)
    for path, expected in run_seal["source_sha256"].items():
        if sha(Path(path)) != expected:
            errors.append(dict(path=path, check="run seal source SHA"))
    output = dict(schema="g32-canonical-development-pooled-review-v1",
                  status="PASS" if not errors else "FAIL", errors=errors,
                  source_reports={n: dict(path=str(ROOT / f"uniform-g32-{n}-canonical-completion-audit.json"),
                                         sha256=sha(ROOT / f"uniform-g32-{n}-canonical-completion-audit.json")) for n in reports},
                  n36_attachment_cross_binding=dict(path=str(receipt), sha256=sha(receipt),
                      producer_receipt_sha256=sha(source_receipt), original_bank_bytes_read=False,
                      all_case_bank_digests_match_completion_guards=True),
                  n36_run_seal=dict(path=str(run_seal_path), sha256=sha(run_seal_path),
                      checked_source_count=len(run_seal["source_sha256"]), all_sources_match=not errors),
                  pooled=pooled, timing_model_solver_training_executed=False, confirmation_banks_accessed=False,
                  interpretation="Neither n36 nor n69 demonstrates a 0.15 original gross-profit ratio advantage. The canonical completion is diagnostic and may violate fixed baseline coordinates; it never substitutes for primary returns. Training-fit limitations remain separate.")
    raw = (json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    target = ROOT / "uniform-g32-canonical-pooled-review.json"
    if target.exists():
        assert target.read_bytes() == raw
    else:
        target.write_bytes(raw)
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
