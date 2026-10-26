"""Exercise actual runner seed expressions without models, solves, or networking.

Production functions are extracted by AST and run with recording stubs. This
keeps the check independent of Torch and avoids copying the faulty expression.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

SUPPLEMENT = Path(__file__).resolve().parents[1]
PARENT = SUPPLEMENT.parent
MAX_HIGHS_SEED = 2**31 - 1


@lru_cache(maxsize=None)
def tree(relative):
    return ast.parse((PARENT / relative).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def compiled_function(relative, name):
    found = [n for n in ast.walk(tree(relative))
             if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise AssertionError(f"Expected exactly one production function {name}")
    prefix = ast.parse("from __future__ import annotations\n").body
    module = ast.fix_missing_locations(ast.Module(body=prefix + found, type_ignores=[]))
    return compile(module, str(PARENT / relative), "exec")


def function(relative, name, globals_):
    exec(compiled_function(relative, name), globals_)
    return globals_[name]


class SeedRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((SUPPLEMENT / "protocol.json").read_text())
        cls.original = json.loads((PARENT / "protocol.json").read_text())
        cls.native = tuple(cls.original["methods"])

    def execute(self, seed, index, method):
        captured = {}

        def native(case, method, models, cost, feature, solver_seed, neural_seed):
            captured.update(solver_seed=solver_seed, neural_seed=neural_seed,
                            method=method, path="native")
            return captured

        def new(method, models, coordinate, case, cost, feature,
                solver_seed, adapter, protocol):
            captured.update(solver_seed=solver_seed, method=method, path="new")
            return captured

        globals_ = dict(args=SimpleNamespace(deadline=None), check_deadline=lambda _: None,
                        case=SimpleNamespace(seed=seed), common=SimpleNamespace(METHODS=self.native),
                        bench=SimpleNamespace(run_method=native), models={}, coordinate=None,
                        new_method=new, adapter=None, protocol=self.protocol)
        run = function("matched-k-supplement-20261010/code/run_supplement.py", "execute", globals_)
        return run(method, [], [], index)

    def assert_highs_seed(self, value):
        self.assertIsInstance(value, int)
        self.assertGreaterEqual(value, 0)
        self.assertLessEqual(value, MAX_HIGHS_SEED)

    def test_actual_warm_call_uses_minus_one(self):
        warm = [n for n in ast.walk(tree("matched-k-supplement-20261010/code/run_supplement.py"))
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "warm" for t in n.targets)]
        self.assertEqual(len(warm), 1)
        call = warm[0].value.elt
        self.assertEqual(call.func.id, "execute")
        self.assertEqual(ast.literal_eval(call.args[-1]), -1)

    def test_all_eleven_routes_warm_and_forty_tests(self):
        self.assertEqual(len(self.protocol["methods"]), 11)
        for seed in self.original["seeds"]:
            for index in [-1] + list(range(self.protocol["confirmation_count"])):
                for method in self.protocol["methods"]:
                    result = self.execute(seed, index, method)
                    self.assert_highs_seed(result["solver_seed"])
                    if "neural_seed" in result:
                        self.assert_highs_seed(result["neural_seed"])

    def test_actual_native_submip_seeds_all_five_candidates(self):
        for seed in self.original["seeds"]:
            for index in [-1] + list(range(self.protocol["confirmation_count"])):
                initial = self.execute(seed, index, "neural_diving_native")["neural_seed"]
                for mode, count in [("native", 5), ("matched", 1)]:
                    captured = []

                    def solve(model, cost, *, fixed, seed, time_limit):
                        captured.append(seed)
                        return SimpleNamespace(seconds=0.0, certified=False)

                    arrays = SimpleNamespace(inf=float("inf"), float64=float,
                                             asarray=lambda value, **_: value,
                                             mean=lambda values: sum(values)/len(values))
                    globals_ = dict(time=SimpleNamespace(perf_counter=lambda: 0.0), np=arrays,
                                    _checked_solution=lambda *args: (None, None),
                                    METHOD_NAME="native", MATCHED_METHOD_NAME="matched")
                    run = function("code/neural_diving_baseline.py", "_run_assignments", globals_)
                    model = SimpleNamespace(graph=SimpleNamespace(integer_indices=[0, 1], device="cpu"))
                    partial = dict(fixed={}, scope_indices=[0, 1], target_K=2,
                                   remaining_scope_count=2, selection="synthetic")
                    run(model, SimpleNamespace(model=None), [], [partial]*count,
                        initial, 0.0, 0.0, 30.0, solve, mode)
                    self.assertEqual(len(captured), count)
                    for value in captured:
                        self.assert_highs_seed(value)

    def test_old_warm_index_reproduces_overflow(self):
        result = self.execute(307, 88000, "neural_diving_native")
        self.assertGreater(result["neural_seed"], MAX_HIGHS_SEED)

    def test_reference_expression_and_frozen_root_lp_seed(self):
        source = tree("matched-k-supplement-20261010/code/run_supplement.py")
        references = [kw.value for n in ast.walk(source) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == "solve"
                      for kw in n.keywords if kw.arg == "seed" and "550000" in ast.unparse(kw.value)]
        self.assertEqual(len(references), 1)
        for seed in self.original["seeds"]:
            for i in range(self.protocol["confirmation_count"]):
                value = eval(compile(ast.Expression(references[0]), "<reference-seed>", "eval"),
                             {"case": SimpleNamespace(seed=seed), "i": i})
                self.assert_highs_seed(value)
        globals_ = dict(hashlib=hashlib, VERSION="independent-confirmation-v2-20261009")
        seed_for = function("code/runner_common.py", "seed_for", globals_)
        for family in self.original["families"]:
            for seed in self.original["seeds"]:
                for amplitude in self.original["amplitudes"]:
                    self.assert_highs_seed(seed_for(f"{family}_seed{seed}_a{amplitude:g}"))


if __name__ == "__main__":
    unittest.main()
