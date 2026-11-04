"""Li fit orchestration with shared external validation; frozen loss/decoder.

Only the old internal split is replaced. The final bank and optimizer see N
training rows only. Native strategy selection and matched inference stay separate.
"""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import torch
import neural_support as data
data.enable_runtime()
from preference_baseline import (PreferenceBaseline, PreferenceConfig, IMPLEMENTATION_VERSION, SOURCE,
    _subset_training_case, _pair_data, _training_hash, _model_hash)

class ExternalValidationPreference(PreferenceBaseline):
    def __init__(self,case,validation_case,config=None):
        super().__init__(case,config=config)
        self.validation_case=validation_case

    def _train_network(self,development_case,bank,pairs,*args,**kwargs):
        # Original October25 grid, clipped only to retained candidate count.
        effective=tuple(dict.fromkeys(min(k,len(pairs["selected"])) for k in (1,3,5,10,30)))
        self.config=replace(self.config,top_k_values=effective)
        return super()._train_network(development_case,bank,pairs,*args,**kwargs)

    def fit(self, output_dir: str | Path | None = None) -> "PreferenceBaseline":
            start = perf_counter()
            destination = Path(output_dir) if output_dir is not None else None
            if destination is not None:
                destination.mkdir(parents=True, exist_ok=True)
            development_start = perf_counter()
            n = len(self.case.train_costs)
            if n < 5:
                raise ValueError("Preference baseline requires at least five training costs for held-out tuning")
            data.validate_training_pair(self.case,self.validation_case)
            validation_count = self.validation_case.n_train
            validation_indices = np.arange(validation_count,dtype=int)
            fit_indices = np.arange(n,dtype=int)
            development_case = _subset_training_case(self.case, fit_indices)
            development_bank = self._bank(development_case)
            development_pairs = _pair_data(development_case, development_bank, self.config,
                destination / "preference-development-pairs.npz" if destination is not None else None)
            validation_costs = np.asarray(self.validation_case.train_costs)
            validation_features = np.asarray(self.validation_case.train_features)
            validation_objectives = np.asarray(self.validation_case.train_objectives)
            tuning = []
            for learning_rate in self.config.learning_rates:
                for lambda1, lambda2 in ((a, b) for a in self.config.lambda1_values for b in self.config.lambda2_values):
                    network, mean, scale, history = self._train_network(development_case, development_bank, development_pairs, learning_rate, lambda1, lambda2, self.config.seed)
                    sanity = self._training_sanity(development_case, development_bank, development_pairs, network, mean, scale, lambda1, lambda2, history)
                    if not sanity["passed"]:
                        tuning.append({"learning_rate": learning_rate, "lambda1": lambda1, "lambda2": lambda2,
                            "top_k": None, "sanity": sanity, "eligible_for_selection": False,
                            "final_training_loss": history[-1]["total"] if history else 0.0})
                        continue
                    for top_k in self.config.top_k_values:
                        results = [self._solve_with(development_bank, development_pairs["selected"], network, mean, scale, c, f, top_k) for c, f in zip(validation_costs, validation_features)]
                        valid = np.array([r["valid"] for r in results], dtype=bool)
                        objectives = np.array([r["objective"] if r["objective"] is not None else np.nan for r in results], dtype=float)
                        tolerances = self.config.recovery_absolute_tolerance + self.config.recovery_relative_tolerance * np.abs(validation_objectives)
                        recovered = valid & (np.abs(objectives - validation_objectives) <= tolerances)
                        gap = np.abs(objectives - validation_objectives) / np.maximum(np.abs(validation_objectives), self.config.denominator_floor)
                        finite_valid_gap = gap[valid & np.isfinite(gap)]
                        mean_gap = float(finite_valid_gap.mean()) if finite_valid_gap.size else float("inf")
                        tuning.append({"learning_rate": learning_rate, "lambda1": lambda1, "lambda2": lambda2, "top_k": top_k, "validation_recovered": int(recovered.sum()), "validation_valid": int(valid.sum()), "validation_count": validation_count, "validation_mean_relative_gap_valid": mean_gap, "final_training_loss": history[-1]["total"] if history else 0.0, "sanity": sanity, "eligible_for_selection": True})
            eligible = [trial for trial in tuning if trial["eligible_for_selection"]]
            if not eligible:
                self.report = {"method": IMPLEMENTATION_VERSION, "configuration": asdict(self.config),
                    "sanity_passed": False, "failure_stage": "all_development_fits_failed_training_sanity",
                    "tuning": tuning, "training_hash": _training_hash(self.case), "model_hash": _model_hash(self.case),
                    "test_information_used": False, "fit_seconds": perf_counter() - start}
                self.report["sanity_gate"] = {"passed": False, "failure_reasons": ["all_development_fits_failed_training_sanity"], "scope": "training_only"}
                if destination is not None:
                    (destination / "preference-training-failed.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")
                raise RuntimeError("All preference development fits failed training sanity; inspect preference-training-failed.json. No benchmark is allowed.")
            best = max(eligible, key=lambda row: (row["validation_recovered"], row["validation_valid"], -row["validation_mean_relative_gap_valid"], -row["top_k"], -row["final_training_loss"]))
            development_selection_seconds = perf_counter() - development_start
            final_bank_start = perf_counter()
            final_training_case = _subset_training_case(self.case, np.arange(n))
            self.bank = self._bank(final_training_case)
            pairs = _pair_data(final_training_case, self.bank, self.config,
                destination / "preference-final-pairs.npz" if destination is not None else None)
            self.selected = pairs["selected"]
            final_bank_preparation_seconds = perf_counter() - final_bank_start
            self.top_k = int(best["top_k"])
            final_optimizer_start = perf_counter()
            final_optimizer_trials, selected_final_trial = [], None
            for learning_rate in self.config.learning_rates:
                trial_start = perf_counter()
                network, mean, scale, trial_history, best_state, trial = self._train_final_network(
                    final_training_case, self.bank, pairs, learning_rate, best["lambda1"], best["lambda2"], self.config.seed)
                diagnostic_start = perf_counter()
                trial["endpoint_cpu_sanity"] = self._training_sanity(final_training_case, self.bank, pairs,
                    network, mean, scale, best["lambda1"], best["lambda2"], trial_history, top_k=self.top_k)
                trial["timing_seconds"]["endpoint_cpu_diagnostics"] = perf_counter() - diagnostic_start
                restore_start = perf_counter()
                network.load_state_dict(best_state)
                trial["timing_seconds"]["retained_state_restore"] = perf_counter() - restore_start
                diagnostic_start = perf_counter()
                trial["retained_cpu_sanity"] = self._training_sanity(final_training_case, self.bank, pairs,
                    network, mean, scale, best["lambda1"], best["lambda2"], trial_history, top_k=self.top_k)
                trial["timing_seconds"]["retained_cpu_diagnostics"] = perf_counter() - diagnostic_start
                trial["retained_objective_vs_rowwise_cpu_loss_delta"] = (
                    trial["retained_cpu_sanity"]["final_total_loss"] - trial["best_full_training_objective"])
                trial["history"] = trial_history
                trial["timing_seconds"]["trial_wall"] = perf_counter() - trial_start
                trial["fit_seconds"] = trial["timing_seconds"]["trial_wall"]
                final_optimizer_trials.append(trial)
                # Readiness and recovery never choose the optimizer/state. A failed
                # minimum-objective state fails the final gate, without fallback.
                if selected_final_trial is None or trial["best_full_training_objective"] < selected_final_trial["best_full_training_objective"]:
                    selected_final_trial = trial
                    self.network, self.mean, self.scale, history = network, mean, scale, trial_history
                del best_state
            final_optimizer_seconds = perf_counter() - final_optimizer_start
            final_sanity = selected_final_trial["retained_cpu_sanity"]
            final_optimizer = {"learning_rate": selected_final_trial["learning_rate"],
                "lambda1": best["lambda1"], "lambda2": best["lambda2"], "top_k": self.top_k,
                "seed": self.config.seed, "retained_epoch": selected_final_trial["best_epoch"],
                "retained_full_training_objective": selected_final_trial["best_full_training_objective"],
                "selection": "minimum full-training objective across retained original-grid states; stable grid-order ties",
                "retained_state_scope": selected_final_trial["best_selection_scope"],
                "scope": "final_training_only", "readiness_used_for_selection": False,
                "recovery_used_for_selection": False, "heldout_used_for_final_learning_rate_selection": False,
                "source_adaptation": True}
            extra_trials = [trial for trial in final_optimizer_trials if trial["learning_rate"] != best["learning_rate"]]
            development_lr_trial = next(trial for trial in final_optimizer_trials if trial["learning_rate"] == best["learning_rate"])
            additional_lr_seconds = sum(trial["timing_seconds"]["trial_wall"] for trial in extra_trials)
            checkpoint_overhead_seconds = sum(development_lr_trial["timing_seconds"][key] for key in
                ("post_update_full_evaluations", "best_state_copies", "retained_state_restore", "retained_cpu_diagnostics"))
            final_optimizer_timing = {"total_wall_seconds": final_optimizer_seconds,
                "original_grid_trial_count": len(final_optimizer_trials), "additional_learning_rate_trial_count": len(extra_trials),
                "additional_learning_rate_trial_wall_seconds": additional_lr_seconds,
                "development_selected_lr_checkpoint_stage_seconds": checkpoint_overhead_seconds,
                "incremental_protocol_stage_seconds": additional_lr_seconds + checkpoint_overhead_seconds,
                "incremental_definition": "Additional LR trials in full plus added post-update evaluation, snapshot, restore and retained-state diagnostics for the development-selected LR; measured stages, not a causal baseline timing comparison"}
            labels = np.asarray(self.bank.labels, dtype=int)
            counts = np.bincount(labels, minlength=len(self.bank.strategies))
            self.report = {"method": "Li2025 preference model reduction (v2 source adaptation)", "implementation_version": IMPLEMENTATION_VERSION, "source": SOURCE, "configuration": asdict(self.config), "selected_hyperparameters": best, "training_hash": _training_hash(self.case), "model_hash": _model_hash(self.case), "training_count": n, "external_validation_indices": validation_indices.tolist(), "training_only_fit_indices": fit_indices.tolist(), "development_candidate_strategies": len(development_bank.strategies), "development_setcover_strategies": len(development_pairs["selected"]), "final_candidate_strategies": len(self.bank.strategies), "final_setcover_strategies": len(self.selected), "selected_strategy_ids": self.selected.tolist(), "setcover_uncovered_training_rows": int((~pairs["edges"][:, self.selected].any(axis=1)).sum()), "training_instance_strategy_pairs": int(pairs["objectives"].size), "selected_training_preference_pairs": int(n * max(len(self.selected) - 1, 0)), "failed_selected_pair_decodes": int(pairs["failed"].sum()), "good_turing_missing_mass_estimate": float((counts == 1).sum() / n), "good_turing_exploration_stop_used": False, "final_training_epochs": self.config.epochs if len(self.selected) > 1 else 0, "tuning": tuning, "history": history, "fit_seconds": perf_counter() - start, "test_information_used": False, "sanity_passed": final_sanity["passed"], "training_sanity": final_sanity, "selection_rule": "minimum normalized original-constraint violation among predicted top-k, Eq. (21)", "b_normalization": "infinity norm of finite canonical constraint RHS, including equality RHS", "decode": "selected inequalities plus permanent original variable bounds and all integer values; dual-certified cached KKT or actual HiGHS reduced LP", "adaptations": ["permanent_original_variable_bounds_unverified_author_domain", "residual_layernorm_token_mlp_attention", "pair_mean_loss_normalization", "independent_lambda2_grid", "uniform_1200_epochs_batch128_decay_every40_and_gradient_clipping", "local_finite_spread_loss_readiness_with_adjacent_accuracy_diagnostic_only", "fixed_training_budget_no_adaptive_good_turing"], "training_device": self.config.device, "inference_device": "cpu", "native_full_integer_strategy": True, "matched_dimension_variant": False, "development_decode_statistics": development_bank.decode_statistics.copy(), "final_decode_statistics": self.bank.decode_statistics.copy()}
            self.report["pair_cache"] = {"development": {key: development_pairs[key] for key in ("cache_reused", "cache_key", "original_decode_statistics")},
                "final": {key: pairs[key] for key in ("cache_reused", "cache_key", "original_decode_statistics")}}
            self.report["sanity_gate"] = final_sanity
            self.report["external_validation_count"] = validation_count
            self.report["external_validation_hash"] = data.training_hash(self.validation_case)
            self.report["validation_labels_in_final_fit"] = False
            self.report["adaptations"].append("separate_external_validation_final_refit_training_rows_only")
            self.report["development_selected_hyperparameters"] = best
            self.report["actual_final_optimizer"] = final_optimizer
            self.report["final_selected_lr"] = final_optimizer["learning_rate"]
            self.report["selected_state_loss"] = final_sanity["final_total_loss"]
            self.report["selected_state_loss_semantics"] = "Fresh rowwise CPU evaluation after restoring the selected actual retained state"
            self.report["history_semantics"] = "Complete terminal pre-update history of the selected LR; its last row is not the retained-state loss"
            self.report["final_training_epochs_semantics"] = "Epochs per nondegenerate LR fit; every configured epoch is executed for each original-grid LR"
            self.report["total_final_training_epochs"] = sum(trial["epochs"] for trial in final_optimizer_trials)
            self.report["final_optimizer_trials"] = final_optimizer_trials
            self.report["final_optimizer_timing"] = final_optimizer_timing
            self.report["development_selection_seconds"] = development_selection_seconds
            self.report["final_bank_preparation_seconds"] = final_bank_preparation_seconds
            self.report["final_optimizer_search_seconds"] = final_optimizer_seconds
            self.report["final_training_optimizer_steps"] = sum(
                trial["epochs"] * int(np.ceil(n / self.config.batch_size)) for trial in final_optimizer_trials)
            self.report["adaptations"].append("uniform_final_original_learning_rate_grid_and_actual_best_post_update_training_state")
            self.report["requested_training_device"] = self.config.device
            self.report["training_device"] = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else ("cpu" if self.config.device == "auto" else self.config.device)
            if output_dir is not None:
                self.save(output_dir)
                np.savez_compressed(Path(output_dir) / "preference-pair-data.npz", p=pairs["p"], d=pairs["d"], edges=pairs["edges"], selected=self.selected, rewards=pairs["rewards"], mean=self.mean, scale=self.scale)
            if not final_sanity["passed"]:
                raise RuntimeError("Final preference fit failed training sanity; saved diagnostic checkpoint is ineligible for benchmarking.")
            return self
