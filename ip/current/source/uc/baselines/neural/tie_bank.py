"""One-solve Li decoder with a certified, fixed representative tie choice.

The reduced inequality LP, its domain and objective are unchanged. A returned
original-feasible point is never replaced. Only a certified optimal reduced-LP
point that violates the original model may be replaced, using this strategy's
fixed first training representative at the same current objective value.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from li_scaling import NativeLPStrategyBank

VERSION = 'li-certified-representative-tie-v2'
ABS_TIE_TOL = 1e-7
REL_TIE_TOL = 1e-10
_STATS = ('raw_original_feasible', 'raw_original_infeasible',
    'tie_candidates_checked', 'tie_representatives_accepted',
    'tie_rejected_certificate', 'tie_rejected_witness_feasibility',
    'tie_rejected_objective')


def source_fingerprint():
    root = Path(__file__).resolve().parent
    return {name:sha256((root/name).read_bytes()).hexdigest()
        for name in ('tie_bank.py', 'tie_api.py')}


def array_hash(array):
    array = np.ascontiguousarray(array, dtype=np.float64)
    digest = sha256(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def constraint_check(model, x, tolerance):
    """Check every retained row/bound and integer domain, without a solve."""
    x = np.asarray(x, dtype=float)
    if x.shape != (model.A.shape[1],) or not np.all(np.isfinite(x)):
        return dict(valid=False, violation=float('inf'), integer_residual=float('inf'))
    ax = np.asarray(model.A @ x).ravel()
    violation = max(0., float(np.max(model.row_lower-ax, initial=0.)),
        float(np.max(ax-model.row_upper, initial=0.)),
        float(np.max(model.col_lower-x, initial=0.)),
        float(np.max(x-model.col_upper, initial=0.)))
    ii = np.flatnonzero(np.asarray(model.integrality))
    integer = float(np.max(np.abs(x[ii]-np.rint(x[ii])), initial=0.))
    return dict(valid=bool(violation <= tolerance and integer <= tolerance),
        violation=violation, integer_residual=integer)


class TieBank(NativeLPStrategyBank):
    def __init__(self, case, config):
        super().__init__(case, config)
        # _pair_data subtracts all statistics from its initial snapshot. Every
        # counter must exist before the first decode, including rejected ties.
        self.decode_statistics.update({name:0 for name in _STATS})
        self.tie_witnesses = []
        for sid, strategy in enumerate(self.strategies):
            index = int(strategy.representative_training_index)
            if not 0 <= index < len(case.train_solutions):
                raise ValueError('Strategy representative index lies outside training data')
            x = np.asarray(case.train_solutions[index], dtype=float).copy()
            original = constraint_check(case.model, x, config.feasibility_tolerance)
            retained = constraint_check(self.reduced_models[sid], x, config.feasibility_tolerance)
            fixed_error = float(np.max(np.abs(x[self.integer_indices]-strategy.integer_values), initial=0.))
            signature = dict(active_rows=np.asarray(strategy.active_rows, dtype=int).tolist(),
                integer_values=np.asarray(strategy.integer_values, dtype=int).tolist())
            key = sha256(json.dumps(signature, sort_keys=True, separators=(',',':')).encode()).hexdigest()
            x.setflags(write=False)
            self.tie_witnesses.append(dict(x=x, training_index=index, x_sha256=array_hash(x),
                strategy_sha256=key, original=original, retained=retained,
                fixed_integer_error=fixed_error,
                eligible=bool(original['valid'] and retained['valid'] and
                    fixed_error <= config.feasibility_tolerance)))
        self.tie_metadata = dict(version=VERSION, absolute_objective_tolerance=ABS_TIE_TOL,
            relative_objective_tolerance=REL_TIE_TOL,
            representative_rule='Fixed first training representative recorded by each extracted strategy',
            witness_count=len(self.tie_witnesses),
            eligible_witness_count=sum(w['eligible'] for w in self.tie_witnesses),
            extra_lp_solves=0, original_certification_claimed=False)

    def decode(self, strategy_index, cost, time_limit=None):
        from adapter import solve
        started = perf_counter()
        self.decode_statistics['calls'] += 1
        try:
            sid = int(strategy_index)
            if sid != strategy_index or not 0 <= sid < len(self.strategies):
                raise ValueError('Unknown strategy index')
            cost = np.asarray(cost, dtype=float)
            if cost.shape != (self.n_variables,) or not np.all(np.isfinite(cost)):
                raise ValueError('A reduced LP needs a finite correctly sized objective')
            remaining = self.config.reduced_lp_time_limit
            if time_limit is not None:
                remaining = min(remaining, max(0., time_limit-(perf_counter()-started)))
            # Exactly the original actual inequality LP, once. No KKT shortcut,
            # second solve, warm start, alternate strategy or full-model solve.
            self.decode_statistics['reduced_lp_solves'] += 1
            solved = solve(self.reduced_models[sid], cost, seed=int(self.case.seed),
                time_limit=remaining, relax_integrality=True)
            reduced_optimal = bool(solved.optimal and solved.valid and solved.x is not None)
            raw_value = None if solved.x is None else float(cost @ solved.x)
            certificate = dict(solver_status=solved.status, optimal=bool(solved.optimal),
                certified=bool(solved.certified), reduced_primal_valid=bool(solved.valid),
                solver_objective=solved.objective, primal_objective=raw_value,
                dual_bound=solved.dual_bound, solver_gap=solved.gap,
                certificate_scope='Same adapter.solve reduced LP; preserve its certification and bound without independent dual reconstruction')
            if reduced_optimal:
                original = self.extracted.check_original(solved.x)
                decoded = dict(x=solved.x, objective=raw_value, valid=original['valid'],
                    status='reduced_lp_highs_optimal' if original['valid'] else 'reduced_lp_highs_optimal_original_infeasible',
                    violation=original['violation'], integer_residual=original['integer_residual'],
                    reduced_optimal=True, decoder='reduced_inequality_lp', reduced_solver_status=solved.status)
                self.decode_statistics['raw_original_feasible' if original['valid'] else 'raw_original_infeasible'] += 1
            else:
                self.decode_statistics['failed_reduced_lp_solves'] += 1
                decoded = dict(x=None, objective=float('inf'), valid=False,
                    status='reduced_lp_'+solved.status.lower().replace(' ','_'),
                    violation=float('inf'), integer_residual=float('inf'), reduced_optimal=False,
                    decoder='reduced_inequality_lp', reduced_solver_status=solved.status)
            tie = dict(version=VERSION, applied=False, reason='raw_return_unchanged',
                raw_status=decoded['status'], raw_objective=decoded['objective'],
                raw_original_valid=bool(decoded['valid']), raw_violation=decoded['violation'],
                raw_integer_residual=decoded['integer_residual'],
                extra_lp_solves=0, certificate=certificate)
            if reduced_optimal and not decoded['valid']:
                self.decode_statistics['tie_candidates_checked'] += 1
                tie['raw_x_sha256'] = array_hash(solved.x)
                tolerance = ABS_TIE_TOL + REL_TIE_TOL*abs(raw_value)
                values = (raw_value, solved.objective, solved.dual_bound)
                cert_ok = (solved.certified and all(v is not None and np.isfinite(v) for v in values)
                    and abs(raw_value-float(solved.objective)) <= tolerance
                    and abs(raw_value-float(solved.dual_bound)) <= tolerance)
                tie.update(objective_tolerance=tolerance, same_lp_certificate_passed=bool(cert_ok))
                if not cert_ok:
                    self.decode_statistics['tie_rejected_certificate'] += 1
                    tie['reason'] = 'same_lp_strict_certificate_failed'
                else:
                    witness = self.tie_witnesses[sid]
                    value = float(cost @ witness['x'])
                    gap = value-raw_value
                    dual_gap = value-float(solved.dual_bound)
                    tie.update(witness_training_index=witness['training_index'],
                        witness_x_sha256=witness['x_sha256'], witness_strategy_sha256=witness['strategy_sha256'],
                        witness_original=witness['original'], witness_retained=witness['retained'],
                        witness_fixed_integer_error=witness['fixed_integer_error'],
                        witness_objective=value, witness_objective_gap=gap, witness_dual_gap=dual_gap)
                    if not witness['eligible']:
                        self.decode_statistics['tie_rejected_witness_feasibility'] += 1
                        tie['reason'] = 'fixed_representative_not_original_and_retained_feasible'
                    elif not np.isfinite(value) or abs(gap) > tolerance or abs(dual_gap) > tolerance:
                        self.decode_statistics['tie_rejected_objective'] += 1
                        tie['reason'] = 'fixed_representative_not_same_current_lp_objective'
                    else:
                        self.decode_statistics['tie_representatives_accepted'] += 1
                        decoded.update(x=witness['x'].copy(), objective=value, valid=True,
                            violation=witness['original']['violation'],
                            integer_residual=witness['original']['integer_residual'],
                            status='reduced_lp_highs_optimal_fixed_representative_tie')
                        tie.update(applied=True, reason='same_optimal_face_fixed_representative')
            decoded.update(strategy_id=sid, negative_dual=0., kkt_objective_gap_bound=float('inf'),
                original_certified=False, reduced_solver_certified=bool(solved.certified),
                reduced_solver_dual_bound=solved.dual_bound, reduced_certificate=certificate,
                representative_tie=tie)
            return decoded
        finally:
            self.decode_statistics['seconds'] += perf_counter()-started
