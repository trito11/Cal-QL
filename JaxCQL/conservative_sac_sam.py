from copy import deepcopy
from functools import partial

from ml_collections import ConfigDict
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax import linen as nn
import optax
from jax.flatten_util import ravel_pytree

from .jax_utils import (
    next_rng, value_and_multi_grad, mse_loss, JaxRNG, wrap_function_with_rng,
    collect_jax_metrics
)
from .model import Scalar, FullyConnectedNetwork, update_target_network


class FullyConnectedValueFunction(nn.Module):
    """Expectile Value Function V_psi(s) for Cal-QL dynamic reference."""
    observation_dim: int
    arch: str = '256-256'
    orthogonal_init: bool = False

    @nn.compact
    def __call__(self, observations):
        x = FullyConnectedNetwork(
            output_dim=1, arch=self.arch, orthogonal_init=self.orthogonal_init
        )(observations)
        return jnp.squeeze(x, -1)

    @nn.nowrap
    def rng_keys(self):
        return ('params',)


def compute_adversarial_action(forward_fn, qf_params, obs, act, rho, clip_min=-1.0, clip_max=1.0):
    """
    Action-Space SAM (Min-Max local ascent on Action Space):
    1. Computes gradient of Q w.r.t action: grad = d(sum(Q(s, a))) / da.
    2. Normalizes by L2 norm across action dim: norm = ||grad||_2 + 1e-8.
    3. Shifts along gradient: delta* = rho * (grad / norm).
    4. Clips to valid action range [clip_min, clip_max].
    5. Returns detached (stop_gradient) adversarial action a*.
    """
    def q_sum(a):
        return jnp.sum(forward_fn(qf_params, obs, a))

    grad = jax.grad(q_sum)(act)
    norm = jnp.linalg.norm(grad, axis=-1, keepdims=True) + 1e-8
    delta_star = rho * (grad / norm)
    a_star = jnp.clip(act + delta_star, clip_min, clip_max)
    return jax.lax.stop_gradient(a_star)


class ConservativeSAC_SAM(object):
    """
    Cal-QL with Dynamic Expectile Value Network & Action-Space SAM:
    1. V_psi(s): In-sample Expectile Value Network replacing static V_ref.
       Trained independently on in-sample transitions (s, a) using asymmetric least squares.
    2. Action-Space SAM: Finds local worst-case OOD point a* in radius rho around policy action
       for the Cal-QL regularizer push-down term.
    3. Protected Bellman target: Evaluated strictly on real dataset transitions (s, a, r, s').
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 1e-4
        config.qf_lr = 3e-4
        config.vf_lr = 3e-4
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = 1.0
        config.cql_temp = 1.0
        config.cql_max_target_backup = True
        config.cql_clip_diff_min = -np.inf
        config.cql_clip_diff_max = np.inf

        # Dynamic Expectile Value Network & Action SAM configurations
        config.tau = 0.8              # Expectile asymmetry weight in (0, 1)
        config.rho = 0.05             # Action perturbation radius rho
        config.use_adv_action = True   # Enable action-space SAM perturbation
        config.use_uncertainty_sam = False # Scale rho by epistemic uncertainty |Q1 - Q2|
        config.uncertainty_scale = 10.0    # Normalization scale for Q-disagreement
        config.use_actor_sam = False       # Enable Actor-level Robust Policy SAM
        config.actor_rho = 0.02            # Perturbation radius for Actor SAM
        config.vf_arch = '256-256'    # Architecture for V_psi(s)
        config.orthogonal_init = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    def __init__(self, config, policy, qf, vf=None):
        self.config = self.get_default_config(config)
        self.policy = policy
        self.qf = qf
        self.observation_dim = policy.observation_dim
        self.action_dim = policy.action_dim

        if vf is not None:
            self.vf = vf
        else:
            self.vf = FullyConnectedValueFunction(
                observation_dim=self.observation_dim,
                arch=self.config.vf_arch,
                orthogonal_init=self.config.orthogonal_init
            )

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        # Policy initialization
        policy_params = self.policy.init(
            next_rng(self.policy.rng_keys()),
            jnp.zeros((10, self.observation_dim))
        )
        self._train_states['policy'] = TrainState.create(
            params=policy_params,
            tx=optimizer_class(self.config.policy_lr),
            apply_fn=None
        )

        # Twin Q-functions initialization
        qf1_params = self.qf.init(
            next_rng(self.qf.rng_keys()),
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim))
        )
        qf2_params = self.qf.init(
            next_rng(self.qf.rng_keys()),
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim))
        )
        self._train_states['qf1'] = TrainState.create(
            params=qf1_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )
        self._train_states['qf2'] = TrainState.create(
            params=qf2_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )
        self._target_qf_params = deepcopy({'qf1': qf1_params, 'qf2': qf2_params})

        # Expectile Value Function V_psi(s) initialization with separate optimizer
        vf_params = self.vf.init(
            next_rng(self.vf.rng_keys()),
            jnp.zeros((10, self.observation_dim))
        )
        self._train_states['vf'] = TrainState.create(
            params=vf_params,
            tx=optimizer_class(self.config.vf_lr),
            apply_fn=None,
        )

        model_keys = ['policy', 'qf1', 'qf2', 'vf']

        # Automatic entropy tuning
        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self._train_states['log_alpha'] = TrainState.create(
                params=self.log_alpha.init(next_rng()),
                tx=optimizer_class(self.config.policy_lr),
                apply_fn=None
            )
            model_keys.append('log_alpha')

        # CQL Lagrange multiplier
        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self._train_states['log_alpha_prime'] = TrainState.create(
                params=self.log_alpha_prime.init(next_rng()),
                tx=optimizer_class(self.config.qf_lr),
                apply_fn=None
            )
            model_keys.append('log_alpha_prime')

        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def train(self, batch, use_cql=True, cql_min_q_weight=5.0, enable_calql=False,
              use_adv_action=None, rho=None, tau=None,
              use_uncertainty_sam=None, uncertainty_scale=None,
              offline_batch_size=-1,
              use_actor_sam=None, actor_rho=None):
        """Public train interface."""
        self._total_steps += 1
        if use_adv_action is None:
            use_adv_action = self.config.use_adv_action
        if rho is None:
            rho = self.config.rho
        if tau is None:
            tau = self.config.tau
        if use_uncertainty_sam is None:
            use_uncertainty_sam = getattr(self.config, 'use_uncertainty_sam', False)
        if uncertainty_scale is None:
            uncertainty_scale = getattr(self.config, 'uncertainty_scale', 10.0)
        if use_actor_sam is None:
            use_actor_sam = getattr(self.config, 'use_actor_sam', False)
        if actor_rho is None:
            actor_rho = getattr(self.config, 'actor_rho', 0.02)

        self._train_states, self._target_qf_params, metrics = self._train_step(
            self._train_states, self._target_qf_params, next_rng(), batch,
            bool(use_cql), jnp.float32(cql_min_q_weight), bool(enable_calql),
            bool(use_adv_action), jnp.float32(rho), jnp.float32(tau),
            bool(use_uncertainty_sam), jnp.float32(uncertainty_scale),
            jnp.int32(offline_batch_size),
            bool(use_actor_sam), jnp.float32(actor_rho)
        )
        return metrics

    @partial(jax.jit, static_argnames=('self', 'use_cql', 'enable_calql', 'use_adv_action', 'use_uncertainty_sam', 'use_actor_sam'))
    def _train_step(self, train_states, target_qf_params, rng, batch,
                    use_cql=True, cql_min_q_weight=5.0, enable_calql=False,
                    use_adv_action=True, rho=0.05, tau=0.8,
                    use_uncertainty_sam=False, uncertainty_scale=10.0,
                    offline_batch_size=-1,
                    use_actor_sam=False, actor_rho=0.02):
        rng_generator = JaxRNG(rng)

        def loss_fn(train_params):
            observations = batch['observations']
            actions = batch['actions']
            rewards = batch['rewards']
            next_observations = batch['next_observations']
            dones = batch['dones']

            loss_collection = {}

            @wrap_function_with_rng(rng_generator())
            def forward_policy(rng, *args, **kwargs):
                return self.policy.apply(
                    *args, **kwargs,
                    rngs=JaxRNG(rng)(self.policy.rng_keys())
                )

            @wrap_function_with_rng(rng_generator())
            def forward_qf(rng, *args, **kwargs):
                return self.qf.apply(
                    *args, **kwargs,
                    rngs=JaxRNG(rng)(self.qf.rng_keys())
                )

            @wrap_function_with_rng(rng_generator())
            def forward_vf(rng, *args, **kwargs):
                return self.vf.apply(
                    *args, **kwargs,
                    rngs=JaxRNG(rng)(self.vf.rng_keys())
                )

            log_pi_data = forward_policy(train_params['policy'], observations, actions, method=self.policy.log_prob)
            new_actions, log_pi = forward_policy(train_params['policy'], observations)

            # ─────────────────────────────────────────────────────────────
            # 1. Automatic Entropy Tuning (Alpha loss)
            # ─────────────────────────────────────────────────────────────
            if self.config.use_automatic_entropy_tuning:
                alpha_loss = -self.log_alpha.apply(train_params['log_alpha']) * (log_pi + self.config.target_entropy).mean()
                loss_collection['log_alpha'] = alpha_loss
                alpha = jnp.exp(self.log_alpha.apply(train_params['log_alpha'])) * self.config.alpha_multiplier
            else:
                alpha_loss = 0.0
                alpha = self.config.alpha_multiplier

            # ─────────────────────────────────────────────────────────────
            # 2. Policy Loss (with optional Actor-SAM Robust Perturbation)
            # ─────────────────────────────────────────────────────────────
            if use_actor_sam:
                def q_actor_sum(a):
                    q1 = forward_qf(train_params['qf1'], observations, a)
                    q2 = forward_qf(train_params['qf2'], observations, a)
                    return jnp.sum(jnp.minimum(q1, q2))

                grad_a = jax.grad(q_actor_sum)(new_actions)
                norm_a = jnp.linalg.norm(grad_a, axis=-1, keepdims=True) + 1e-8
                delta_worst = -actor_rho * (grad_a / norm_a)
                delta_worst = jax.lax.stop_gradient(delta_worst)
                a_robust = jnp.clip(new_actions + delta_worst, -1.0, 1.0)

                q_new_actions = jnp.minimum(
                    forward_qf(train_params['qf1'], observations, a_robust),
                    forward_qf(train_params['qf2'], observations, a_robust),
                )
                actor_adv_norm = jnp.mean(jnp.linalg.norm(delta_worst, axis=-1))
            else:
                q_new_actions = jnp.minimum(
                    forward_qf(train_params['qf1'], observations, new_actions),
                    forward_qf(train_params['qf2'], observations, new_actions),
                )
                actor_adv_norm = 0.0

            policy_loss = (alpha * log_pi - q_new_actions).mean()
            loss_collection['policy'] = policy_loss

            # ─────────────────────────────────────────────────────────────
            # 3. Dynamic Expectile Value Function Loss (loss_V)
            # In-sample regression on replay buffer transitions (s, a)
            # ─────────────────────────────────────────────────────────────
            target_q1_in_sample = forward_qf(target_qf_params['qf1'], observations, actions)
            target_q2_in_sample = forward_qf(target_qf_params['qf2'], observations, actions)
            target_q_in_sample_min = jax.lax.stop_gradient(
                jnp.minimum(target_q1_in_sample, target_q2_in_sample)
            )

            v_pred = forward_vf(train_params['vf'], observations)
            diff_v = target_q_in_sample_min - v_pred
            weight_v = jnp.where(diff_v > 0, tau, 1.0 - tau)
            sq_err_v = weight_v * jnp.square(diff_v)

            # Protected V_psi: if offline_batch_size > 0, only train on offline dataset slice
            batch_indices = jnp.arange(observations.shape[0])
            mask_v = jnp.where((offline_batch_size > 0) & (batch_indices >= offline_batch_size), 0.0, 1.0)
            loss_v = jnp.sum(mask_v * sq_err_v) / jnp.maximum(jnp.sum(mask_v), 1.0)
            loss_collection['vf'] = loss_v

            # ─────────────────────────────────────────────────────────────
            # 4. Critic Q-function Loss (Bellman MSE with real dataset pairs)
            # Bellman targets are strictly computed on dataset transitions (s, a, r, s')
            # ─────────────────────────────────────────────────────────────
            q1_pred = forward_qf(train_params['qf1'], observations, actions)
            q2_pred = forward_qf(train_params['qf2'], observations, actions)

            if self.config.cql_max_target_backup:
                new_next_actions, next_log_pi = forward_policy(
                    train_params['policy'], next_observations, repeat=self.config.cql_n_actions
                )
                target_q_values = jnp.minimum(
                    forward_qf(target_qf_params['qf1'], next_observations, new_next_actions),
                    forward_qf(target_qf_params['qf2'], next_observations, new_next_actions),
                )
                max_target_indices = jnp.expand_dims(jnp.argmax(target_q_values, axis=-1), axis=-1)
                target_q_values = jnp.take_along_axis(target_q_values, max_target_indices, axis=-1).squeeze(-1)
                next_log_pi = jnp.take_along_axis(next_log_pi, max_target_indices, axis=-1).squeeze(-1)
            else:
                new_next_actions, next_log_pi = forward_policy(
                    train_params['policy'], next_observations
                )
                target_q_values = jnp.minimum(
                    forward_qf(target_qf_params['qf1'], next_observations, new_next_actions),
                    forward_qf(target_qf_params['qf2'], next_observations, new_next_actions),
                )

            if self.config.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            td_target = jax.lax.stop_gradient(
                rewards + (1.0 - dones) * self.config.discount * target_q_values
            )
            qf1_bellman_loss = mse_loss(q1_pred, td_target)
            qf2_bellman_loss = mse_loss(q2_pred, td_target)

            # ─────────────────────────────────────────────────────────────
            # 5. Cal-QL Regularizer with Action-Space SAM Perturbation
            # ─────────────────────────────────────────────────────────────
            adv_action_norm = 0.0
            if use_cql:
                batch_size = actions.shape[0]
                cql_random_actions = jax.random.uniform(
                    rng_generator(), shape=(batch_size, self.config.cql_n_actions, self.action_dim),
                    minval=-1.0, maxval=1.0
                )

                cql_current_actions, cql_current_log_pis = forward_policy(
                    train_params['policy'], observations, repeat=self.config.cql_n_actions,
                )
                cql_next_actions, cql_next_log_pis = forward_policy(
                    train_params['policy'], next_observations, repeat=self.config.cql_n_actions,
                )

                # Action-Space SAM: compute locally worst-case perturbed actions a*
                if use_adv_action:
                    q1_curr_nom = forward_qf(train_params['qf1'], observations, cql_current_actions)
                    q2_curr_nom = forward_qf(train_params['qf2'], observations, cql_current_actions)

                    if use_uncertainty_sam:
                        disagreement = jnp.abs(q1_curr_nom - q2_curr_nom)  # (batch_size, n_actions)
                        uncertainty_factor = jnp.clip(disagreement / jnp.maximum(uncertainty_scale, 1e-6), 0.0, 1.0)
                        rho_eff = jnp.expand_dims(rho * uncertainty_factor, -1)  # (batch_size, n_actions, 1)
                    else:
                        disagreement = jnp.zeros_like(q1_curr_nom)
                        rho_eff = rho

                    cql_act_star_q1 = compute_adversarial_action(
                        forward_qf, train_params['qf1'], observations, cql_current_actions, rho_eff
                    )
                    cql_act_star_q2 = compute_adversarial_action(
                        forward_qf, train_params['qf2'], observations, cql_current_actions, rho_eff
                    )
                    adv_action_norm = jnp.mean(jnp.linalg.norm(cql_act_star_q1 - cql_current_actions, axis=-1))

                    cql_q1_current_actions = forward_qf(train_params['qf1'], observations, cql_act_star_q1)
                    cql_q2_current_actions = forward_qf(train_params['qf2'], observations, cql_act_star_q2)
                else:
                    disagreement = jnp.zeros((batch_size, self.config.cql_n_actions))
                    rho_eff = jnp.zeros(())
                    adv_action_norm = 0.0
                    cql_q1_current_actions = forward_qf(train_params['qf1'], observations, cql_current_actions)
                    cql_q2_current_actions = forward_qf(train_params['qf2'], observations, cql_current_actions)

                cql_q1_rand = forward_qf(train_params['qf1'], observations, cql_random_actions)
                cql_q2_rand = forward_qf(train_params['qf2'], observations, cql_random_actions)
                cql_q1_next_actions = forward_qf(train_params['qf1'], observations, cql_next_actions)
                cql_q2_next_actions = forward_qf(train_params['qf2'], observations, cql_next_actions)

                # Cal-QL Lower Bounding: using dynamic Expectile V_psi(s)
                # (replacing static MC returns or frozen V_ref)
                v_stop = jax.lax.stop_gradient(v_pred)
                lower_bounds = jnp.repeat(jnp.expand_dims(v_stop, 1), cql_q1_current_actions.shape[1], axis=1)

                num_vals = jnp.sum(lower_bounds == lower_bounds)
                bound_rate_cql_q1_current_actions = jnp.sum(cql_q1_current_actions < lower_bounds) / num_vals
                bound_rate_cql_q2_current_actions = jnp.sum(cql_q2_current_actions < lower_bounds) / num_vals
                bound_rate_cql_q1_next_actions = jnp.sum(cql_q1_next_actions < lower_bounds) / num_vals
                bound_rate_ql_q2_next_actions = jnp.sum(cql_q2_next_actions < lower_bounds) / num_vals

                if enable_calql:
                    cql_q1_current_actions = jnp.maximum(cql_q1_current_actions, lower_bounds)
                    cql_q2_current_actions = jnp.maximum(cql_q2_current_actions, lower_bounds)
                    cql_q1_next_actions = jnp.maximum(cql_q1_next_actions, lower_bounds)
                    cql_q2_next_actions = jnp.maximum(cql_q2_next_actions, lower_bounds)

                cql_cat_q1 = jnp.concatenate(
                    [cql_q1_rand, jnp.expand_dims(q1_pred, 1), cql_q1_next_actions, cql_q1_current_actions], axis=1
                )
                cql_cat_q2 = jnp.concatenate(
                    [cql_q2_rand, jnp.expand_dims(q2_pred, 1), cql_q2_next_actions, cql_q2_current_actions], axis=1
                )
                cql_std_q1 = jnp.std(cql_cat_q1, axis=1)
                cql_std_q2 = jnp.std(cql_cat_q2, axis=1)

                if self.config.cql_importance_sample:
                    random_density = np.log(0.5 ** self.action_dim)
                    cql_cat_q1 = jnp.concatenate(
                        [cql_q1_rand - random_density,
                         cql_q1_next_actions - cql_next_log_pis,
                         cql_q1_current_actions - cql_current_log_pis],
                        axis=1
                    )
                    cql_cat_q2 = jnp.concatenate(
                        [cql_q2_rand - random_density,
                         cql_q2_next_actions - cql_next_log_pis,
                         cql_q2_current_actions - cql_current_log_pis],
                        axis=1
                    )

                cql_qf1_ood = (
                    jax.scipy.special.logsumexp(cql_cat_q1 / self.config.cql_temp, axis=1)
                    * self.config.cql_temp
                )
                cql_qf2_ood = (
                    jax.scipy.special.logsumexp(cql_cat_q2 / self.config.cql_temp, axis=1)
                    * self.config.cql_temp
                )

                # Push-down on OOD / perturbed policy actions, push-up on buffer actions
                cql_qf1_diff = jnp.clip(
                    cql_qf1_ood - q1_pred,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()
                cql_qf2_diff = jnp.clip(
                    cql_qf2_ood - q2_pred,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()

                if self.config.cql_lagrange:
                    alpha_prime = jnp.clip(
                        jnp.exp(self.log_alpha_prime.apply(train_params['log_alpha_prime'])),
                        a_min=0.0, a_max=1000000.0
                    )
                    cql_min_qf1_loss = alpha_prime * cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                    cql_min_qf2_loss = alpha_prime * cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)
                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
                    loss_collection['log_alpha_prime'] = alpha_prime_loss
                else:
                    cql_min_qf1_loss = cql_qf1_diff * cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * cql_min_q_weight
                    alpha_prime_loss = 0.0
                    alpha_prime = 0.0

                qf1_loss = qf1_bellman_loss + cql_min_qf1_loss
                qf2_loss = qf2_bellman_loss + cql_min_qf2_loss
            else:
                qf1_loss = qf1_bellman_loss
                qf2_loss = qf2_bellman_loss

            loss_collection['qf1'] = qf1_loss
            loss_collection['qf2'] = qf2_loss

            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params)

        policy_loss_gradient = jnp.linalg.norm(ravel_pytree(grads[self.model_keys.index("policy")]['policy'])[0])
        qf1_loss_gradient = jnp.linalg.norm(ravel_pytree(grads[self.model_keys.index("qf1")]['qf1'])[0])
        qf2_loss_gradient = jnp.linalg.norm(ravel_pytree(grads[self.model_keys.index("qf2")]['qf2'])[0])
        vf_loss_gradient = jnp.linalg.norm(ravel_pytree(grads[self.model_keys.index("vf")]['vf'])[0])

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        new_target_qf_params = {}
        new_target_qf_params['qf1'] = update_target_network(
            new_train_states['qf1'].params, target_qf_params['qf1'],
            self.config.soft_target_update_rate
        )
        new_target_qf_params['qf2'] = update_target_network(
            new_train_states['qf2'].params, target_qf_params['qf2'],
            self.config.soft_target_update_rate
        )

        metrics = collect_jax_metrics(
            aux_values,
            ['log_pi', 'policy_loss', 'qf1_loss', 'qf2_loss', 'loss_v', 'v_pred',
             'alpha_loss', 'alpha', 'q1_pred', 'q2_pred', 'target_q_values', 'adv_action_norm']
        )
        metrics.update(
            policy_loss_gradient=policy_loss_gradient,
            qf1_loss_gradient=qf1_loss_gradient,
            qf2_loss_gradient=qf2_loss_gradient,
            vf_loss_gradient=vf_loss_gradient,
            use_cql=int(use_cql),
            enable_calql=int(enable_calql),
            cql_min_q_weight=cql_min_q_weight,
            rho=jnp.mean(rho),
            tau=tau,
            use_adv_action=int(use_adv_action),
            use_uncertainty_sam=int(use_uncertainty_sam),
            uncertainty_disagreement_mean=jnp.mean(aux_values['disagreement']) if use_cql else jnp.zeros(()),
            rho_effective_mean=jnp.mean(aux_values['rho_eff']) if (use_cql and use_adv_action) else jnp.zeros(()),
            v_offline_mask_ratio=jnp.mean(aux_values['mask_v']),
            use_actor_sam=int(use_actor_sam),
            actor_rho=actor_rho,
            actor_adv_norm=aux_values['actor_adv_norm'],
        )

        if use_cql:
            metrics.update(collect_jax_metrics(
                aux_values,
                ['cql_std_q1', 'cql_std_q2', 'cql_q1_rand', 'cql_q2_rand',
                 'cql_qf1_diff', 'cql_qf2_diff', 'cql_min_qf1_loss',
                 'cql_min_qf2_loss', 'cql_q1_current_actions', 'cql_q2_current_actions',
                 'cql_q1_next_actions', 'cql_q2_next_actions', 'alpha_prime',
                 'alpha_prime_loss', 'qf1_bellman_loss', 'qf2_bellman_loss',
                 'bound_rate_cql_q1_current_actions', 'bound_rate_cql_q2_current_actions',
                 'bound_rate_cql_q1_next_actions', 'bound_rate_ql_q2_next_actions', 'log_pi_data'],
                'cql'
            ))

        return new_train_states, new_target_qf_params, metrics

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps
