"""
Cal-QL with Dynamic Expectile Value Network & Action-Space SAM
=============================================================
Main experiment entry point for Cal-QL refactored:
- Expectile Value Network V(s) replacing static V_ref (IQL style).
- Action-space SAM perturbation a* in Cal-QL regularizer.
- Strict Bellman target protection with real dataset pairs.
"""

import numpy as np
import gym
import d4rl
import time

import absl.app
import absl.flags

from .conservative_sac_sam import ConservativeSAC_SAM
from .replay_buffer import (
    subsample_batch, concatenate_batches,
    get_hand_dataset_with_mc_calculation, ReplayBuffer
)
from .jax_utils import batch_to_jax
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import TrajSampler
from .utils import (
    Timer, define_flags_with_default, set_random_seed,
    get_user_flags, prefix_metrics, WandBLogger
)
from viskit.logging import logger, setup_logger


FLAGS_DEF = define_flags_with_default(
    env='pen-binary-v0',
    seed=42,
    save_model=False,
    batch_size=256,

    reward_scale=10.0,
    reward_bias=5.0,
    clip_action=0.99999,

    policy_arch='512-512',
    qf_arch='512-512-512',
    orthogonal_init=True,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_train_step_per_epoch_offline=1000,
    n_pretrain_epochs=20,
    offline_eval_every_n_epoch=2,

    max_online_env_steps=3e4,
    online_eval_every_n_env_steps=1000,

    eval_n_trajs=20,
    replay_buffer_size=1000000,
    mixing_ratio=0.5,
    use_cql=True,
    online_use_cql=True,
    cql_min_q_weight=1.0,
    cql_min_q_weight_online=-1.0,
    enable_calql=True,

    n_online_traj_per_epoch=1,
    online_utd_ratio=1,

    # ── Cal-QL with Expectile Value Network & Action SAM flags ────────────
    tau=0.8,                   # Expectile loss parameter tau in (0, 1)
    rho=0.05,                  # Adversarial action perturbation radius rho (offline)
    rho_online=-1.0,           # SAM radius in online phase (-1.0 to use rho)
    sam_rho_decay=False,       # Linearly decay rho during online phase down to rho_online (or 0)
    sam_start_epoch=0,         # Start applying SAM at or after this offline epoch (e.g. 10 for warmup)
    use_adv_action=True,       # Enable adversarial action perturbation
    use_uncertainty_sam=False, # Scale rho by epistemic uncertainty |Q1 - Q2|
    uncertainty_scale=10.0,    # Normalization scale for Q-disagreement
    v_offline_only_online=False, # Train V_psi strictly on offline dataset transitions during online
    run_id=0,
    run_label='CalQL-Expectile-ActionSAM',

    # Backwards compatibility flags
    use_sam=False,
    sam_target='both',
    sam_start_ratio=0.8,
    sam_rho_schedule='fixed',
    sam_rho_max=0.05,
    sam_rho_min=0.005,
    sam_gamma=0.01,

    cql=ConservativeSAC_SAM.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    variant['run_label'] = FLAGS.run_label
    variant['tau'] = FLAGS.tau
    variant['rho'] = FLAGS.rho
    variant['rho_online'] = FLAGS.rho_online
    variant['sam_rho_decay'] = FLAGS.sam_rho_decay
    variant['sam_start_epoch'] = FLAGS.sam_start_epoch
    variant['use_adv_action'] = FLAGS.use_adv_action
    variant['use_uncertainty_sam'] = FLAGS.use_uncertainty_sam
    variant['uncertainty_scale'] = FLAGS.uncertainty_scale
    variant['v_offline_only_online'] = FLAGS.v_offline_only_online

    # ── WandB logger setup ────────────────────────────────────────────────
    import wandb, uuid, os
    from .utils import WandBLogger

    _orig_init = WandBLogger.__init__

    def _patched_init(self, config, variant):
        from copy import copy
        import tempfile
        from socket import gethostname

        self.config = self.get_default_config(config)
        if self.config.experiment_id is None:
            self.config.experiment_id = ''
        if self.config.prefix != '':
            self.config.project = '{}--{}'.format(self.config.prefix, self.config.project)
        if self.config.output_dir == '':
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(
                self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)
        if 'hostname' not in self._variant:
            self._variant['hostname'] = gethostname()

        try:
            from .wandb_config import get_wandb_config
            wandb_cfg = get_wandb_config()
            os.environ['WANDB_API_KEY'] = wandb_cfg['WANDB_API_KEY']
            os.environ['WANDB_USER_EMAIL'] = wandb_cfg['WANDB_EMAIL']
            os.environ['WANDB_USERNAME'] = wandb_cfg['WANDB_USERNAME']
            os.environ['WANDB_MODE'] = 'run'
        except Exception:
            print('No wandb_config.py — running offline')
            os.environ['WANDB_MODE'] = 'run'
            self.config.online = False

        run_name = variant.get('run_label', 'unknown')
        env_name = variant.get('env', 'unknown')

        self.run = wandb.init(
            reinit=True,
            name=run_name,
            group=env_name,
            job_type=run_name,
            tags=[env_name, run_name, f"tau={FLAGS.tau}", f"rho={FLAGS.rho}"],
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            id=self.config.experiment_id + uuid.uuid4().hex,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method='thread',
                _disable_stats=True,
            ),
            mode='online' if self.config.online else 'offline',
            entity=self.config.entity,
        )

    WandBLogger.__init__ = _patched_init

    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    print(f"\n{'='*70}")
    print(f"  Cal-QL: Dynamic Expectile Value Network & Action-Space SAM")
    print(f"  run_label={FLAGS.run_label}  seed={FLAGS.seed}")
    print(f"  tau={FLAGS.tau}  rho={FLAGS.rho}  use_adv_action={FLAGS.use_adv_action}")
    print(f"  enable_calql={FLAGS.enable_calql}  use_cql={FLAGS.use_cql}")
    print(f"{'='*70}\n")

    # ── Dataset + Env ─────────────────────────────────────────────────────
    import mj_envs
    dataset = get_hand_dataset_with_mc_calculation(
        FLAGS.env, gamma=FLAGS.cql.discount,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        clip_action=FLAGS.clip_action
    )
    assert dataset['next_observations'].shape == dataset['observations'].shape

    set_random_seed(FLAGS.seed)
    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, use_goal=True,
                                gamma=FLAGS.cql.discount)
    train_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, use_goal=True,
                                use_mc=True, gamma=FLAGS.cql.discount,
                                reward_scale=FLAGS.reward_scale,
                                reward_bias=FLAGS.reward_bias)
    replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)

    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim      = eval_sampler.env.action_space.shape[0]

    policy = TanhGaussianPolicy(
        observation_dim, action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset)
    qf = FullyConnectedQFunction(
        observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(
            eval_sampler.env.action_space.shape).item()

    # Pass tau, rho, use_adv_action, and anti-dip flags into cql config
    FLAGS.cql.tau = FLAGS.tau
    FLAGS.cql.rho = FLAGS.rho
    FLAGS.cql.use_adv_action = FLAGS.use_adv_action
    FLAGS.cql.use_uncertainty_sam = FLAGS.use_uncertainty_sam
    FLAGS.cql.uncertainty_scale = FLAGS.uncertainty_scale

    sac = ConservativeSAC_SAM(FLAGS.cql, policy, qf)
    sampler_policy = SamplerPolicy(sac.policy, sac.train_params['policy'])

    # ── Training loop ─────────────────────────────────────────────────────
    viskit_metrics = {}
    n_train_step_per_epoch = FLAGS.n_train_step_per_epoch_offline
    cql_min_q_weight = FLAGS.cql_min_q_weight
    enable_calql     = FLAGS.enable_calql
    use_cql          = FLAGS.use_cql
    mixing_ratio     = FLAGS.mixing_ratio

    total_grad_steps      = 0
    is_online             = False
    online_eval_counter   = -1
    online_rollout_timer  = None
    train_timer           = None
    epoch                 = 0
    train_metrics         = None
    expl_metrics          = None
    online_td_losses      = []

    while True:
        metrics = {'epoch': epoch}

        if epoch == FLAGS.n_pretrain_epochs:
            is_online = True
            if FLAGS.cql_min_q_weight_online >= 0:
                cql_min_q_weight = FLAGS.cql_min_q_weight_online
            if not FLAGS.online_use_cql and use_cql:
                use_cql = False

        do_eval = (
            epoch == 0
            or (not is_online and epoch % FLAGS.offline_eval_every_n_epoch == 0)
            or epoch == FLAGS.n_pretrain_epochs
            or (is_online and
                replay_buffer.total_steps // FLAGS.online_eval_every_n_env_steps
                > online_eval_counter)
            or replay_buffer.total_steps >= FLAGS.max_online_env_steps
        )

        with Timer() as eval_timer:
            if do_eval:
                trajs = eval_sampler.sample(
                    sampler_policy.update_params(sac.train_params['policy']),
                    FLAGS.eval_n_trajs, deterministic=True)
                metrics['evaluation/average_return'] = np.mean(
                    [np.sum(t['rewards']) for t in trajs])
                metrics['evaluation/average_traj_length'] = np.mean(
                    [len(t['rewards']) for t in trajs])
                metrics['evaluation/goal_achieved_rate'] = np.mean(
                    [1 in t['goal_achieved'] for t in trajs])
                if is_online:
                    online_eval_counter = (replay_buffer.total_steps
                                           // FLAGS.online_eval_every_n_env_steps)

        metrics.update({
            'run_id':   FLAGS.run_id,
            'run_label': FLAGS.run_label,
            'grad_steps': total_grad_steps,
            'epoch': epoch,
            'online_rollout_time': 0 if online_rollout_timer is None else online_rollout_timer(),
            'train_time':          0 if train_timer is None else train_timer(),
            'eval_time':           eval_timer(),
            'epoch_time':          eval_timer() if train_timer is None
                                   else train_timer() + eval_timer(),
            'mixing_ratio': mixing_ratio,
        })
        if is_online:
            metrics['env_steps'] = replay_buffer.total_steps

        if is_online and train_metrics is not None:
            td_key = 'sac/qf1_bellman_loss'
            if td_key in train_metrics:
                online_td_losses.append(float(train_metrics[td_key]))
                metrics['diagnostic/td_loss_peak'] = max(online_td_losses)

        if train_metrics is not None:
            metrics.update(train_metrics)
        if expl_metrics is not None:
            metrics.update(expl_metrics)

        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

        if replay_buffer.total_steps >= FLAGS.max_online_env_steps:
            print(f"\n[{FLAGS.run_label}] FINISHED — "
                  f"TD-Loss Peak={max(online_td_losses):.4f}" if online_td_losses
                  else f"\n[{FLAGS.run_label}] FINISHED")
            break

        # Online rollout
        with Timer() as online_rollout_timer:
            if is_online:
                trajs = train_sampler.sample(
                    sampler_policy.update_params(sac.train_params['policy']),
                    n_trajs=FLAGS.n_online_traj_per_epoch,
                    deterministic=False, replay_buffer=replay_buffer)
                expl_metrics = {
                    'exploration/average_return': np.mean(
                        [np.sum(t['rewards']) for t in trajs]),
                    'exploration/average_traj_length': np.mean(
                        [len(t['rewards']) for t in trajs]),
                    'exploration/goal_achieved_rate': np.mean(
                        [1 in t['goal_achieved'] for t in trajs]),
                }

        if train_timer is None:
            print(f"[{FLAGS.run_label}] JIT compiling...")

        with Timer() as train_timer:
            if epoch >= FLAGS.n_pretrain_epochs and FLAGS.online_utd_ratio > 0:
                n_train_step_per_epoch = (
                    np.sum([len(t['rewards']) for t in trajs])
                    * FLAGS.online_utd_ratio)

            if FLAGS.mixing_ratio >= 0:
                mixing_ratio = FLAGS.mixing_ratio
            else:
                mixing_ratio = (dataset['rewards'].shape[0]
                                / (dataset['rewards'].shape[0]
                                   + replay_buffer.total_steps))
            batch_size_offline = int(FLAGS.batch_size * mixing_ratio)
            batch_size_online  = FLAGS.batch_size - batch_size_offline

            # Determine dynamic SAM parameters and protected V status
            if is_online:
                target_rho = FLAGS.rho_online if FLAGS.rho_online >= 0.0 else FLAGS.rho
                if FLAGS.sam_rho_decay:
                    progress = min(1.0, replay_buffer.total_steps / float(FLAGS.max_online_env_steps))
                    current_rho = float(FLAGS.rho - progress * (FLAGS.rho - target_rho))
                else:
                    current_rho = float(target_rho)
                current_use_adv = FLAGS.use_adv_action and (current_rho > 0.0)
                offline_batch_size_arg = batch_size_offline if FLAGS.v_offline_only_online else -1
            else:
                current_use_adv = FLAGS.use_adv_action and (epoch >= FLAGS.sam_start_epoch)
                current_rho = float(FLAGS.rho)
                offline_batch_size_arg = -1

            step_metrics_accum = []
            for _ in range(int(n_train_step_per_epoch)):
                if is_online:
                    batch = batch_to_jax(concatenate_batches([
                        subsample_batch(dataset, batch_size_offline),
                        replay_buffer.sample(batch_size_online),
                    ]))
                else:
                    batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))

                step_m = prefix_metrics(
                    sac.train(
                        batch,
                        use_cql=use_cql,
                        cql_min_q_weight=cql_min_q_weight,
                        enable_calql=enable_calql,
                        use_adv_action=current_use_adv,
                        rho=current_rho,
                        tau=FLAGS.tau,
                        use_uncertainty_sam=FLAGS.use_uncertainty_sam,
                        uncertainty_scale=FLAGS.uncertainty_scale,
                        offline_batch_size=offline_batch_size_arg,
                    ),
                    'sac'
                )

                step_metrics_accum.append(step_m)
                total_grad_steps += 1

            # Average numeric metrics over the epoch
            train_metrics = {}
            for key in step_metrics_accum[-1]:
                vals = [m[key] for m in step_metrics_accum
                        if not isinstance(m[key], str)]
                if vals:
                    train_metrics[key] = float(np.mean(vals))

        epoch += 1


if __name__ == '__main__':
    absl.app.run(main)
