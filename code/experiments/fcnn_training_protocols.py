#!/usr/bin/env python3
"""FCNN-c training protocols used by the archived main-text experiments."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch


CORE_CODE = Path(__file__).resolve().parent.parent / "core"
if str(CORE_CODE) not in sys.path:
    sys.path.insert(0, str(CORE_CODE))

import final_projection_figure_suite as fps  # noqa: E402
import iwata_sakaue_pelp_projection_compare as iw  # noqa: E402


_LEGACY_FUNCTIONS = {
    "train_signed": fps.train_signed_costonly_model,
    "train_direct": fps.train_implicit_projection_model,
    "initialize": fps.initialize_signed_costonly_model,
}
_PLAIN_RANDOM_FUNCTIONS: dict[str, object] | None = None


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and Torch without selecting on test performance."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _capture_functions() -> dict[str, object]:
    return {
        "train_signed": fps.train_signed_costonly_model,
        "train_direct": fps.train_implicit_projection_model,
        "initialize": fps.initialize_signed_costonly_model,
    }


def _install_functions(functions: dict[str, object]) -> None:
    fps.train_signed_costonly_model = functions["train_signed"]
    fps.train_implicit_projection_model = functions["train_direct"]
    fps.initialize_signed_costonly_model = functions["initialize"]


def use_legacy_cost_pca_protocol() -> None:
    """Restore the archived cost-PCA initialized FCNN-c implementation."""
    _install_functions(_LEGACY_FUNCTIONS)


def use_plain_random_protocol() -> None:
    """Use the archived plain-random FCNN-c implementation."""
    global _PLAIN_RANDOM_FUNCTIONS
    if _PLAIN_RANDOM_FUNCTIONS is None:
        _install_plain_random_functions()
        _PLAIN_RANDOM_FUNCTIONS = _capture_functions()
    else:
        _install_functions(_PLAIN_RANDOM_FUNCTIONS)


def _install_plain_random_functions() -> None:
    def train_signed_costonly_model(
        model,
        train,
        val,
        N,
        x_ref,
        epochs,
        batch_size,
        lr,
        device,
        seed,
        patience=120,
        verbose=False,
    ):
        rng = np.random.default_rng(seed)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        history = {"train_ratio": [], "val_ratio": []}
        best_state = copy.deepcopy(model.state_dict())
        best_val = (
            fps.evaluate_signed_costonly(model, val, N, x_ref, device)[
                "objective_ratio_mean"
            ]
            if len(val) > 0
            else -float("inf")
        )
        bad_epochs = 0
        for ep in range(1, int(epochs) + 1):
            order = rng.permutation(len(train))
            model.train()
            batch_terms = []
            train_ratios = []
            opt.zero_grad(set_to_none=True)

            for pos, idx in enumerate(order, start=1):
                inst = train[int(idx)]
                g_t = torch.as_tensor(
                    inst.g_reward,
                    dtype=torch.float32,
                    device=device,
                )
                P_t = model(g_t)
                P_np = P_t.detach().cpu().numpy().astype(float)
                success, obj, _runtime, y, lam, _x_std = (
                    fps.solve_signed_nullspace_lp_max(inst, P_np, N, x_ref)
                )
                if not success or y is None or lam is None:
                    continue
                train_ratios.append(
                    fps.objective_ratio(obj, inst.full_objective)
                )
                grad_vec = (
                    np.asarray(inst.g_reward, dtype=float)
                    + np.asarray(N, dtype=float).T @ np.asarray(lam, dtype=float)
                )
                grad_np = np.outer(grad_vec, np.asarray(y, dtype=float))
                grad_np /= max(abs(float(inst.full_objective)), 1e-8)
                grad_t = torch.as_tensor(
                    grad_np,
                    dtype=P_t.dtype,
                    device=device,
                )
                batch_terms.append((P_t * grad_t).sum())
                if len(batch_terms) == int(batch_size) or pos == len(order):
                    if batch_terms:
                        loss = -torch.stack(batch_terms).mean()
                        loss.backward()
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        batch_terms = []

            val_ratio = fps.evaluate_signed_costonly(
                model,
                val,
                N,
                x_ref,
                device,
            )["objective_ratio_mean"]
            train_ratio = (
                float(np.nanmean(train_ratios))
                if train_ratios
                else float("nan")
            )
            history["train_ratio"].append(train_ratio)
            history["val_ratio"].append(val_ratio)
            tol = max(1e-12, 1e-4 * max(abs(best_val), 1e-12))
            if val_ratio > best_val + tol:
                best_val = val_ratio
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
            if verbose and (
                ep == 1 or ep % 50 == 0 or ep == int(epochs)
            ):
                print(
                    f"signed cost-only epoch {ep:04d}/{epochs} | "
                    f"train={train_ratio:.6g} val={val_ratio:.6g} "
                    f"best={best_val:.6g}",
                    flush=True,
                )
            if int(patience) > 0 and bad_epochs >= int(patience):
                break
        model.load_state_dict(best_state)
        return model, history

    def train_implicit_projection_model(
        model,
        train,
        val,
        epochs,
        batch_size,
        lr,
        device,
        seed,
        patience=120,
        verbose=False,
    ):
        rng = np.random.default_rng(seed)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        history = {"train_ratio": [], "val_ratio": []}
        best_state = copy.deepcopy(model.state_dict())
        best_val = (
            iw.evaluate_learned_projector(model, val, device)[
                "objective_ratio_mean"
            ]
            if len(val) > 0
            else -float("inf")
        )
        bad_epochs = 0
        for ep in range(1, int(epochs) + 1):
            order = rng.permutation(len(train))
            model.train()
            batch_terms = []
            train_ratios = []
            opt.zero_grad(set_to_none=True)

            for pos, idx in enumerate(order, start=1):
                inst = train[int(idx)]
                c_t, A_t, b_t = iw.to_torch_instance(inst, device)
                P_t = model(c_t, A_t, b_t)
                P_np = P_t.detach().cpu().numpy().astype(float)
                sol, y = iw.solve_projected_y_max(inst, P_np)
                if not sol.success or y is None:
                    continue
                lam = sol.dual_ub_for_max
                if lam is None:
                    lam = np.zeros(inst.n_cons, dtype=float)
                train_ratios.append(
                    float(
                        sol.obj
                        / max(abs(float(inst.full_obj or 0.0)), 1e-8)
                    )
                )
                grad_np = iw.implicit_grad_P_with_y(
                    inst,
                    y,
                    lam,
                    sol.dual_eq_for_max,
                    True,
                )
                grad_t = torch.as_tensor(
                    grad_np,
                    dtype=P_t.dtype,
                    device=device,
                )
                batch_terms.append((P_t * grad_t).sum())
                if len(batch_terms) == int(batch_size) or pos == len(order):
                    if batch_terms:
                        loss = -torch.stack(batch_terms).mean()
                        loss.backward()
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        batch_terms = []

            val_ratio = iw.evaluate_learned_projector(
                model,
                val,
                device,
            )["objective_ratio_mean"]
            train_ratio = (
                float(np.nanmean(train_ratios))
                if train_ratios
                else float("nan")
            )
            history["train_ratio"].append(train_ratio)
            history["val_ratio"].append(val_ratio)
            tol = max(1e-12, 1e-4 * max(abs(best_val), 1e-12))
            if val_ratio > best_val + tol:
                best_val = val_ratio
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
            if verbose and (
                ep == 1 or ep % 50 == 0 or ep == int(epochs)
            ):
                print(
                    f"cost-only direct epoch {ep:04d}/{epochs} | "
                    f"train={train_ratio:.6g} val={val_ratio:.6g} "
                    f"best={best_val:.6g}",
                    flush=True,
                )
            if int(patience) > 0 and bad_epochs >= int(patience):
                break
        model.load_state_dict(best_state)
        return model, history

    fps.train_signed_costonly_model = train_signed_costonly_model
    fps.train_implicit_projection_model = train_implicit_projection_model
    fps.initialize_signed_costonly_model = lambda model, P0: None
