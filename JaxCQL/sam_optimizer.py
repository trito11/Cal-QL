"""
SAM (Sharpness-Aware Minimization) Optimizer for JAX/Optax
=============================================================

Implements the Min-Max formulation:
    min_θ max_{||ε||₂ ≤ ρ} L(θ + ε)

Two-pass algorithm:
  Step 1 (first_step):  ε*(θ) = ρ · ∇L(θ) / ||∇L(θ)||₂
                        θ_perturbed = θ + ε*
  Step 2 (second_step): θ ← θ - η · ∇L(θ_perturbed)   (restore then update)

Three ρ scheduling modes:
  - 'fixed'             : ρ_t = ρ₀
  - 'cosine_decay'      : cosine decay from ρ_max → ρ_min over [start_step, end_step]
  - 'grad_norm_adaptive': ρ_t = clip(γ · ||∇L||₂, ρ_min, ρ_max)
"""

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from jax.flatten_util import ravel_pytree
from typing import Literal
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ρ Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class SAMRhoScheduler:
    """
    Computes the SAM perturbation radius ρ_t based on the chosen schedule.

    Args:
        mode         : 'fixed' | 'cosine_decay' | 'grad_norm_adaptive'
        rho_fixed    : constant ρ for 'fixed' mode
        rho_max      : upper bound for ρ
        rho_min      : lower bound for ρ
        gamma        : scale factor for grad_norm_adaptive mode
        start_step   : step at which cosine decay begins  (absolute step count)
        end_step     : step at which cosine decay ends    (absolute step count)
    """

    def __init__(
        self,
        mode: Literal['fixed', 'cosine_decay', 'grad_norm_adaptive'] = 'fixed',
        rho_fixed: float = 0.05,
        rho_max: float = 0.05,
        rho_min: float = 0.005,
        gamma: float = 0.01,
        start_step: int = 0,
        end_step: int = 1,
    ):
        self.mode = mode
        self.rho_fixed = rho_fixed
        self.rho_max = rho_max
        self.rho_min = rho_min
        self.gamma = gamma
        self.start_step = start_step
        self.end_step = max(end_step, start_step + 1)

    def get_rho(self, current_step: int, grad_norm: float = 1.0) -> float:
        """Return scalar ρ_t given current training step."""
        if self.mode == 'fixed':
            return self.rho_fixed

        elif self.mode == 'cosine_decay':
            # Cosine decay from rho_max → rho_min over [start_step, end_step]
            t = max(0, current_step - self.start_step)
            T = self.end_step - self.start_step
            cos_val = 0.5 * (1.0 + np.cos(np.pi * t / T))
            return self.rho_min + (self.rho_max - self.rho_min) * cos_val

        elif self.mode == 'grad_norm_adaptive':
            # Adaptive: clip(γ · ||∇L||₂, ρ_min, ρ_max)
            rho = self.gamma * float(grad_norm)
            return float(np.clip(rho, self.rho_min, self.rho_max))

        else:
            raise ValueError(f"Unknown SAM rho mode: {self.mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Core SAM perturbation helpers (pure JAX, jit-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def compute_perturbation(grads_pytree, rho: float):
    """
    Compute SAM perturbation:
        ε* = ρ · grads / ||grads||₂

    Works on arbitrary pytrees (nested dicts of jnp arrays).

    Returns:
        perturbation : pytree of same shape as grads_pytree
        grad_norm    : scalar L2 norm of the flat gradient vector
    """
    flat_grads, _ = ravel_pytree(grads_pytree)
    grad_norm = jnp.linalg.norm(flat_grads)
    # Avoid division by zero
    safe_norm = jnp.maximum(grad_norm, 1e-12)
    perturbation = jax.tree_util.tree_map(
        lambda g: rho * g / safe_norm, grads_pytree
    )
    return perturbation, grad_norm


def apply_perturbation(params_pytree, perturbation_pytree):
    """θ_perturbed = θ + ε*"""
    return jax.tree_util.tree_map(
        lambda p, e: p + e, params_pytree, perturbation_pytree
    )


def remove_perturbation(params_pytree, perturbation_pytree):
    """θ_restored = θ_perturbed - ε*"""
    return jax.tree_util.tree_map(
        lambda p, e: p - e, params_pytree, perturbation_pytree
    )


# ─────────────────────────────────────────────────────────────────────────────
# SAM-aware gradient update for a single Flax TrainState
# ─────────────────────────────────────────────────────────────────────────────

def sam_update_train_state(
    train_state: TrainState,
    first_grads,           # gradients at θ (first pass)
    second_grads,          # gradients at θ + ε* (second pass)
):
    """
    Apply a standard optax gradient update using second_grads
    (SAM always uses the second-pass gradients for the parameter update,
     but the train_state optimizer step count is only incremented once).

    Returns the updated TrainState.
    """
    return train_state.apply_gradients(grads=second_grads)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: measure sharpness  L(θ+ε*) - L(θ)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sharpness(loss_at_theta: float, loss_at_perturbed: float) -> float:
    """Sharpness proxy = L(θ+ε*) - L(θ).  Higher → sharper minimum."""
    return float(loss_at_perturbed) - float(loss_at_theta)
