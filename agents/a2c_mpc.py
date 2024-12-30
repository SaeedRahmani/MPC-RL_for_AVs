import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor, set_random_seed
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr
from stable_baselines3.common.utils import check_for_correct_spaces, get_system_info
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env
from stable_baselines3.a2c import A2C

from agents.pure_mpc_saeed import PureMPC_Agent

SelfA2C = TypeVar("SelfA2C", bound="A2C")

class A2C_MPC(A2C):
    """
    Advantage Actor-Critic algorithm (A2C) adapted for integration with MPC.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
    :param n_steps: The number of steps to run for each environment per update
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: Smoothing term for RMSProp optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
    :param normalize_advantage: Whether to normalize or not the advantage
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        mpcrl_cfg: Dict,
        version: str,
        pure_mpc_cfg: Dict,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rms_prop_eps=rms_prop_eps,
            use_rms_prop=use_rms_prop,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.normalize_advantage = normalize_advantage
        self.version = version
        self.mpc_agent = PureMPC_Agent(
            env=self.env.envs[0],
            cfg=pure_mpc_cfg,
        )

        if _init_setup_model:
            self._setup_model()

    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill the RolloutBuffer.

        The RL agent generates actions, which are processed by the MPC to output
        the final actions sent to the environment.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Get final actions from MPC based on RL outputs
            if self.version == "v0":
                ref_speed = actions  # RL provides reference speed
                weights_from_RL = None
            else:
                ref_speed = None
                weights_from_RL = actions  # RL provides weights for cost function

            mpc_action = self.mpc_agent.predict(
                obs=np.squeeze(self._last_obs, axis=0),
                return_numpy=True,
                ref_speed=ref_speed,
                weights_from_RL=weights_from_RL,
            )
            mpc_action = mpc_action.reshape((1, 2))  # Acceleration, steering angle

            new_obs, rewards, dones, infos = env.step(mpc_action)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Perform a single update step using the gathered rollout buffer.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        entropy_losses = []
        pg_losses, value_losses = [], []

        for rollout_data in self.rollout_buffer.get(self.n_steps):
            actions = rollout_data.actions

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss
            value_loss = F.mse_loss(rollout_data.returns, values)
            value_losses.append(value_loss.item())

            # Entropy loss
            entropy_loss = -th.mean(entropy) if entropy is not None else -log_prob.mean()
            entropy_losses.append(entropy_loss.item())

            # Total loss
            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Logging
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "A2C",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfA2C:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    @classmethod
    def load(
        cls,
        path,
        mpcrl_cfg,
        version,
        pure_mpc_cfg,
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> "A2C_MPC":
        """
        Load the model from a zip-file and reinitialize the MPC agent.
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs. "
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Adjust the environment action space to match the model
            env.action_space = data["action_space"]

            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])

            # Reset the environment if required
            if force_reset and data is not None:
                data["_last_obs"] = None

        model = cls(
            mpcrl_cfg,
            version,
            pure_mpc_cfg,
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,
        )

        # Load parameters and set model attributes
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to set model parameters: {e}. Ensure the saved model is compatible with this class."
            )

        # Reinitialize MPC agent
        model.mpc_agent = PureMPC_Agent(
            env=model.env.envs[0],
            cfg=pure_mpc_cfg,
        )
        model.version = version

        return model


# import warnings
# from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

# import numpy as np
# import torch as th
# from gymnasium import spaces
# from torch.nn import functional as F

# from stable_baselines3.common.buffers import RolloutBuffer
# from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
# from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
# from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.utils import explained_variance, obs_as_tensor, set_random_seed
# from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr
# from stable_baselines3.common.utils import check_for_correct_spaces, get_system_info
# from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env
# from stable_baselines3.a2c import A2C
# from agents.pure_mpc_saeed import PureMPC_Agent

# SelfA2C = TypeVar("SelfA2C", bound="A2C")

# class A2C_MPC(A2C):
#     """
#     Advantage Actor-Critic algorithm (A2C) adapted for integration with MPC.

#     :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
#     :param env: The environment to learn from (if registered in Gym, can be str)
#     :param learning_rate: The learning rate, it can be a function
#     :param n_steps: The number of steps to run for each environment per update
#     :param gamma: Discount factor
#     :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
#     :param ent_coef: Entropy coefficient for the loss calculation
#     :param vf_coef: Value function coefficient for the loss calculation
#     :param max_grad_norm: The maximum value for the gradient clipping
#     :param rms_prop_eps: Smoothing term for RMSProp optimizer
#     :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
#     :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
#     :param normalize_advantage: Whether to normalize or not the advantage
#     :param tensorboard_log: the log location for tensorboard (if None, no logging)
#     :param policy_kwargs: additional arguments to be passed to the policy on creation
#     :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
#     :param seed: Seed for the pseudo random generators
#     :param device: Device (cpu, cuda, ...) on which the code should be run.
#     """

#     policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
#         "MlpPolicy": ActorCriticPolicy,
#         "CnnPolicy": ActorCriticCnnPolicy,
#         "MultiInputPolicy": MultiInputActorCriticPolicy,
#     }

#     def __init__(
#         self,
#         mpcrl_cfg: Dict,
#         version: str,
#         pure_mpc_cfg: Dict,
#         policy: Union[str, Type[ActorCriticPolicy]],
#         env: Union[GymEnv, str],
#         learning_rate: Union[float, Schedule] = 7e-4,
#         n_steps: int = 5,
#         gamma: float = 0.99,
#         gae_lambda: float = 1.0,
#         ent_coef: float = 0.0,
#         vf_coef: float = 0.5,
#         max_grad_norm: float = 0.5,
#         rms_prop_eps: float = 1e-5,
#         use_rms_prop: bool = True,
#         use_sde: bool = False,
#         sde_sample_freq: int = -1,
#         normalize_advantage: bool = False,
#         tensorboard_log: Optional[str] = None,
#         policy_kwargs: Optional[Dict[str, Any]] = None,
#         verbose: int = 0,
#         seed: Optional[int] = None,
#         device: Union[th.device, str] = "auto",
#         _init_setup_model: bool = True,
#     ):
#         super().__init__(
#             policy,
#             env,
#             learning_rate=learning_rate,
#             n_steps=n_steps,
#             gamma=gamma,
#             gae_lambda=gae_lambda,
#             ent_coef=ent_coef,
#             vf_coef=vf_coef,
#             max_grad_norm=max_grad_norm,
#             rms_prop_eps=rms_prop_eps,
#             use_rms_prop=use_rms_prop,
#             use_sde=use_sde,
#             sde_sample_freq=sde_sample_freq,
#             tensorboard_log=tensorboard_log,
#             policy_kwargs=policy_kwargs,
#             verbose=verbose,
#             seed=seed,
#             device=device,
#             _init_setup_model=False,
#         )

#         self.normalize_advantage = normalize_advantage
#         self.version = version
#         self.mpc_agent = PureMPC_Agent(
#             env=self.env.envs[0],
#             cfg=pure_mpc_cfg,
#         )

#         if _init_setup_model:
#             self._setup_model()

#     def collect_rollouts(
#         self,
#         env,
#         callback: BaseCallback,
#         rollout_buffer: RolloutBuffer,
#         n_rollout_steps: int,
#     ) -> bool:
#         """
#         Collect experiences using the current policy and fill the RolloutBuffer.

#         The RL agent generates actions, which are processed by the MPC to output
#         the final actions sent to the environment.
#         """
#         assert self._last_obs is not None, "No previous observation was provided"
#         self.policy.set_training_mode(False)

#         n_steps = 0
#         rollout_buffer.reset()
#         callback.on_rollout_start()

#         while n_steps < n_rollout_steps:
#             with th.no_grad():
#                 obs_tensor = obs_as_tensor(self._last_obs, self.device)
#                 actions, values, log_probs = self.policy(obs_tensor)
#             actions = actions.cpu().numpy()

#             # Get final actions from MPC based on RL outputs
#             if self.version == "v0":
#                 ref_speed = actions  # RL provides reference speed
#                 weights_from_RL = None
#             else:
#                 ref_speed = None
#                 weights_from_RL = actions  # RL provides weights for cost function

#             mpc_action = self.mpc_agent.predict(
#                 obs=np.squeeze(self._last_obs, axis=0),
#                 return_numpy=True,
#                 ref_speed=ref_speed,
#                 weights_from_RL=weights_from_RL,
#             )
#             mpc_action = mpc_action.reshape((1, 2))  # Acceleration, steering angle

#             new_obs, rewards, dones, infos = env.step(mpc_action)
#             self.num_timesteps += env.num_envs

#             callback.update_locals(locals())
#             if not callback.on_step():
#                 return False

#             self._update_info_buffer(infos, dones)
#             n_steps += 1

#             rollout_buffer.add(
#                 self._last_obs,
#                 actions,
#                 rewards,
#                 self._last_episode_starts,
#                 values,
#                 log_probs,
#             )
#             self._last_obs = new_obs
#             self._last_episode_starts = dones

#         with th.no_grad():
#             values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

#         rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
#         callback.on_rollout_end()

#         return True

#     @classmethod
#     def load(
#         cls,
#         path,
#         mpcrl_cfg,
#         version,
#         pure_mpc_cfg,
#         env: Optional[GymEnv] = None,
#         device: Union[th.device, str] = "auto",
#         custom_objects: Optional[Dict[str, Any]] = None,
#         print_system_info: bool = False,
#         force_reset: bool = True,
#         **kwargs,
#     ) -> "A2C_MPC":
#         """
#         Load the model from a zip-file and reinitialize the MPC agent.
#         """
#         if print_system_info:
#             print("== CURRENT SYSTEM INFO ==")
#             get_system_info()

#         data, params, pytorch_variables = load_from_zip_file(
#             path,
#             device=device,
#             custom_objects=custom_objects,
#             print_system_info=print_system_info,
#         )

#         assert data is not None, "No data found in the saved file"
#         assert params is not None, "No params found in the saved file"

#         # Remove stored device information and replace with ours
#         if "policy_kwargs" in data:
#             if "device" in data["policy_kwargs"]:
#                 del data["policy_kwargs"]["device"]

#         if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
#             raise ValueError(
#                 f"The specified policy kwargs do not equal the stored policy kwargs. "
#                 f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
#             )

#         if "observation_space" not in data or "action_space" not in data:
#             raise KeyError("The observation_space and action_space were not given, can't verify new environments")

#         # Gym -> Gymnasium space conversion
#         for key in {"observation_space", "action_space"}:
#             data[key] = _convert_space(data[key])

#         if env is not None:
#             # Check if given env is valid
#             check_for_correct_spaces(env, data["observation_space"], data["action_space"])

#             # Adjust the action space to match the model
#             env.action_space = data["action_space"]

#             # Reset the environment if required
#             if force_reset and data is not None:
#                 data["_last_obs"] = None

#         model = cls(
#             mpcrl_cfg,
#             version,
#             pure_mpc_cfg,
#             policy=data["policy_class"],
#             env=env,
#             device=device,
#             _init_setup_model=False,
#         )

#         # Load parameters and set model attributes
#         model.__dict__.update(data)
#         model.__dict__.update(kwargs)
#         model._setup_model()

#         try:
#             model.set_parameters(params, exact_match=True, device=device)
#         except RuntimeError as e:
#             raise RuntimeError(
#                 f"Failed to set model parameters: {e}. Ensure the saved model is compatible with this class."
#             )

#         # Reinitialize MPC agent
#         model.mpc_agent = PureMPC_Agent(
#             env=model.env.envs[0],
#             cfg=pure_mpc_cfg,
#         )
#         model.version = version

#         return model


# # import warnings
# # from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

# # import numpy as np
# # import torch as th
# # from gymnasium import spaces
# # from torch.nn import functional as F

# # from stable_baselines3.common.buffers import RolloutBuffer
# # from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
# # from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
# # from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
# # from stable_baselines3.common.callbacks import BaseCallback
# # from stable_baselines3.common.utils import explained_variance, obs_as_tensor, set_random_seed
# # from stable_baselines3.a2c import A2C
# # from agents.pure_mpc_saeed import PureMPC_Agent

# # SelfA2C = TypeVar("SelfA2C", bound="A2C")

# # class A2C_MPC(A2C):
# #     """
# #     Advantage Actor-Critic algorithm (A2C) adapted for integration with MPC.

# #     :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
# #     :param env: The environment to learn from (if registered in Gym, can be str)
# #     :param learning_rate: The learning rate, it can be a function
# #     :param n_steps: The number of steps to run for each environment per update
# #     :param gamma: Discount factor
# #     :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
# #     :param ent_coef: Entropy coefficient for the loss calculation
# #     :param vf_coef: Value function coefficient for the loss calculation
# #     :param max_grad_norm: The maximum value for the gradient clipping
# #     :param rms_prop_eps: Smoothing term for RMSProp optimizer
# #     :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
# #     :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
# #     :param normalize_advantage: Whether to normalize or not the advantage
# #     :param tensorboard_log: the log location for tensorboard (if None, no logging)
# #     :param policy_kwargs: additional arguments to be passed to the policy on creation
# #     :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
# #     :param seed: Seed for the pseudo random generators
# #     :param device: Device (cpu, cuda, ...) on which the code should be run.
# #     """

# #     policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
# #         "MlpPolicy": ActorCriticPolicy,
# #         "CnnPolicy": ActorCriticCnnPolicy,
# #         "MultiInputPolicy": MultiInputActorCriticPolicy,
# #     }

# #     def __init__(
# #         self,
# #         mpcrl_cfg: Dict,
# #         version: str,
# #         pure_mpc_cfg: Dict,
# #         policy: Union[str, Type[ActorCriticPolicy]],
# #         env: Union[GymEnv, str],
# #         learning_rate: Union[float, Schedule] = 7e-4,
# #         n_steps: int = 5,
# #         gamma: float = 0.99,
# #         gae_lambda: float = 1.0,
# #         ent_coef: float = 0.0,
# #         vf_coef: float = 0.5,
# #         max_grad_norm: float = 0.5,
# #         rms_prop_eps: float = 1e-5,
# #         use_rms_prop: bool = True,
# #         use_sde: bool = False,
# #         sde_sample_freq: int = -1,
# #         normalize_advantage: bool = False,
# #         tensorboard_log: Optional[str] = None,
# #         policy_kwargs: Optional[Dict[str, Any]] = None,
# #         verbose: int = 0,
# #         seed: Optional[int] = None,
# #         device: Union[th.device, str] = "auto",
# #         _init_setup_model: bool = True,
# #     ):
# #         super().__init__(
# #             policy,
# #             env,
# #             learning_rate=learning_rate,
# #             n_steps=n_steps,
# #             gamma=gamma,
# #             gae_lambda=gae_lambda,
# #             ent_coef=ent_coef,
# #             vf_coef=vf_coef,
# #             max_grad_norm=max_grad_norm,
# #             rms_prop_eps=rms_prop_eps,
# #             use_rms_prop=use_rms_prop,
# #             use_sde=use_sde,
# #             sde_sample_freq=sde_sample_freq,
# #             tensorboard_log=tensorboard_log,
# #             policy_kwargs=policy_kwargs,
# #             verbose=verbose,
# #             seed=seed,
# #             device=device,
# #             _init_setup_model=False,
# #         )

# #         self.normalize_advantage = normalize_advantage
# #         self.version = version
# #         self.mpc_agent = PureMPC_Agent(
# #             env=self.env.envs[0],
# #             cfg=pure_mpc_cfg,
# #         )

# #         if _init_setup_model:
# #             self._setup_model()

# #     def collect_rollouts(
# #         self,
# #         env,
# #         callback: BaseCallback,
# #         rollout_buffer: RolloutBuffer,
# #         n_rollout_steps: int,
# #     ) -> bool:
# #         """
# #         Collect experiences using the current policy and fill the RolloutBuffer.

# #         The RL agent generates actions, which are processed by the MPC to output
# #         the final actions sent to the environment.
# #         """
# #         assert self._last_obs is not None, "No previous observation was provided"
# #         self.policy.set_training_mode(False)

# #         n_steps = 0
# #         rollout_buffer.reset()
# #         callback.on_rollout_start()

# #         while n_steps < n_rollout_steps:
# #             with th.no_grad():
# #                 obs_tensor = obs_as_tensor(self._last_obs, self.device)
# #                 actions, values, log_probs = self.policy(obs_tensor)
# #             actions = actions.cpu().numpy()

# #             # Get final actions from MPC based on RL outputs
# #             if self.version == "v0":
# #                 ref_speed = actions  # RL provides reference speed
# #                 weights_from_RL = None
# #             else:
# #                 ref_speed = None
# #                 weights_from_RL = actions  # RL provides weights for cost function

# #             mpc_action = self.mpc_agent.predict(
# #                 obs=np.squeeze(self._last_obs, axis=0),
# #                 return_numpy=True,
# #                 ref_speed=ref_speed,
# #                 weights_from_RL=weights_from_RL,
# #             )
# #             mpc_action = mpc_action.reshape((1, 2))  # Acceleration, steering angle

# #             new_obs, rewards, dones, infos = env.step(mpc_action)
# #             self.num_timesteps += env.num_envs

# #             callback.update_locals(locals())
# #             if not callback.on_step():
# #                 return False

# #             self._update_info_buffer(infos, dones)
# #             n_steps += 1

# #             rollout_buffer.add(
# #                 self._last_obs,
# #                 actions,
# #                 rewards,
# #                 self._last_episode_starts,
# #                 values,
# #                 log_probs,
# #             )
# #             self._last_obs = new_obs
# #             self._last_episode_starts = dones

# #         with th.no_grad():
# #             values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

# #         rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
# #         callback.on_rollout_end()

# #         return True

# #     @classmethod
# #     def load(
# #         cls,
# #         path,
# #         mpcrl_cfg,
# #         version,
# #         pure_mpc_cfg,
# #         env: Optional[GymEnv] = None,
# #         device: Union[th.device, str] = "auto",
# #         **kwargs,
# #     ) -> "A2C_MPC":
# #         """
# #         Load the model from a zip-file and reinitialize the MPC agent.
# #         """
# #         model = super(A2C_MPC, cls).load(
# #             path,
# #             env=env,
# #             device=device,
# #             **kwargs,
# #         )
# #         model.mpc_agent = PureMPC_Agent(
# #             env=model.env.envs[0],
# #             cfg=pure_mpc_cfg,
# #         )
# #         model.version = version
# #         return model

# # # import warnings
# # # from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

# # # import numpy as np
# # # import torch as th
# # # from gymnasium import spaces
# # # from torch.nn import functional as F

# # # from stable_baselines3.common.buffers import RolloutBuffer
# # # from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
# # # from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
# # # from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
# # # from stable_baselines3.common.callbacks import BaseCallback
# # # from stable_baselines3.common.utils import explained_variance, obs_as_tensor, set_random_seed
# # # from stable_baselines3.a2c import A2C
# # # from agents.pure_mpc_saeed import PureMPC_Agent

# # # SelfA2C = TypeVar("SelfA2C", bound="A2C")

# # # class A2C_MPC(A2C):
# # #     """
# # #     Advantage Actor-Critic algorithm (A2C) adapted for integration with MPC.

# # #     :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
# # #     :param env: The environment to learn from (if registered in Gym, can be str)
# # #     :param learning_rate: The learning rate, it can be a function
# # #     :param n_steps: The number of steps to run for each environment per update
# # #     :param gamma: Discount factor
# # #     :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
# # #     :param ent_coef: Entropy coefficient for the loss calculation
# # #     :param vf_coef: Value function coefficient for the loss calculation
# # #     :param max_grad_norm: The maximum value for the gradient clipping
# # #     :param rms_prop_eps: Smoothing term for RMSProp optimizer
# # #     :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
# # #     :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
# # #     :param normalize_advantage: Whether to normalize or not the advantage
# # #     :param tensorboard_log: the log location for tensorboard (if None, no logging)
# # #     :param policy_kwargs: additional arguments to be passed to the policy on creation
# # #     :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
# # #     :param seed: Seed for the pseudo random generators
# # #     :param device: Device (cpu, cuda, ...) on which the code should be run.
# # #     """

# # #     policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
# # #         "MlpPolicy": ActorCriticPolicy,
# # #         "CnnPolicy": ActorCriticCnnPolicy,
# # #         "MultiInputPolicy": MultiInputActorCriticPolicy,
# # #     }

# # #     def __init__(
# # #         self,
# # #         mpcrl_cfg: Dict,
# # #         version: str,
# # #         pure_mpc_cfg: Dict,
# # #         policy: Union[str, Type[ActorCriticPolicy]],
# # #         env: Union[GymEnv, str],
# # #         learning_rate: Union[float, Schedule] = 7e-4,
# # #         n_steps: int = 5,
# # #         gamma: float = 0.99,
# # #         gae_lambda: float = 1.0,
# # #         ent_coef: float = 0.0,
# # #         vf_coef: float = 0.5,
# # #         max_grad_norm: float = 0.5,
# # #         rms_prop_eps: float = 1e-5,
# # #         use_rms_prop: bool = True,
# # #         use_sde: bool = False,
# # #         sde_sample_freq: int = -1,
# # #         normalize_advantage: bool = False,
# # #         tensorboard_log: Optional[str] = None,
# # #         policy_kwargs: Optional[Dict[str, Any]] = None,
# # #         verbose: int = 0,
# # #         seed: Optional[int] = None,
# # #         device: Union[th.device, str] = "auto",
# # #         _init_setup_model: bool = True,
# # #     ):
# # #         super().__init__(
# # #             policy,
# # #             env,
# # #             learning_rate=learning_rate,
# # #             n_steps=n_steps,
# # #             gamma=gamma,
# # #             gae_lambda=gae_lambda,
# # #             ent_coef=ent_coef,
# # #             vf_coef=vf_coef,
# # #             max_grad_norm=max_grad_norm,
# # #             rms_prop_eps=rms_prop_eps,
# # #             use_rms_prop=use_rms_prop,
# # #             use_sde=use_sde,
# # #             sde_sample_freq=sde_sample_freq,
# # #             tensorboard_log=tensorboard_log,
# # #             policy_kwargs=policy_kwargs,
# # #             verbose=verbose,
# # #             seed=seed,
# # #             device=device,
# # #             _init_setup_model=False,
# # #         )

# # #         self.normalize_advantage = normalize_advantage
# # #         self.version = version
# # #         self.mpc_agent = PureMPC_Agent(
# # #             env=self.env.envs[0],
# # #             cfg=pure_mpc_cfg,
# # #         )

# # #         if _init_setup_model:
# # #             self._setup_model()

# # #     def collect_rollouts(
# # #         self,
# # #         env,
# # #         callback: BaseCallback,
# # #         rollout_buffer: RolloutBuffer,
# # #         n_rollout_steps: int,
# # #     ) -> bool:
# # #         """
# # #         Collect experiences and fill the rollout buffer.
# # #         """
# # #         assert self._last_obs is not None, "No previous observation was provided"
# # #         self.policy.set_training_mode(False)

# # #         n_steps = 0
# # #         rollout_buffer.reset()

# # #         callback.on_rollout_start()

# # #         while n_steps < n_rollout_steps:
# # #             with th.no_grad():
# # #                 obs_tensor = obs_as_tensor(self._last_obs, self.device)
# # #                 actions, values, log_probs = self.policy(obs_tensor)
# # #             actions = actions.cpu().numpy()

# # #             # Rescale and perform action
# # #             clipped_actions = actions
# # #             if isinstance(self.action_space, spaces.Box):
# # #                 clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

# # #             # Get MPC action based on RL output
# # #             if self.version == "v0":
# # #                 ref_speed = clipped_actions
# # #                 weights_from_RL = None
# # #             else:
# # #                 ref_speed = None
# # #                 weights_from_RL = clipped_actions

# # #             mpc_action = self.mpc_agent.predict(
# # #                 obs=np.squeeze(self._last_obs, axis=0),
# # #                 return_numpy=True,
# # #                 weights_from_RL=weights_from_RL,
# # #                 ref_speed=ref_speed,
# # #             )
# # #             mpc_action = mpc_action.reshape((1, 2))

# # #             new_obs, rewards, dones, infos = env.step(mpc_action)
# # #             self.num_timesteps += env.num_envs

# # #             callback.update_locals(locals())
# # #             if not callback.on_step():
# # #                 return False

# # #             self._update_info_buffer(infos, dones)
# # #             n_steps += 1

# # #             rollout_buffer.add(
# # #                 self._last_obs,
# # #                 actions,
# # #                 rewards,
# # #                 self._last_episode_starts,
# # #                 values,
# # #                 log_probs,
# # #             )
# # #             self._last_obs = new_obs
# # #             self._last_episode_starts = dones

# # #         with th.no_grad():
# # #             values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

# # #         rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
# # #         callback.on_rollout_end()

# # #         return True

# # #     @classmethod
# # #     def load(
# # #         cls,
# # #         path,
# # #         mpcrl_cfg,
# # #         version,
# # #         pure_mpc_cfg,
# # #         env: Optional[GymEnv] = None,
# # #         device: Union[th.device, str] = "auto",
# # #         **kwargs,
# # #     ) -> "A2C_MPC":
# # #         """ Load the model from a zip-file."""
# # #         model = super(A2C_MPC, cls).load(
# # #             path,
# # #             env=env,
# # #             device=device,
# # #             **kwargs,
# # #         )
# # #         model.mpc_agent = PureMPC_Agent(
# # #             env=model.env.envs[0],
# # #             cfg=pure_mpc_cfg,
# # #         )
# # #         model.version = version
# # #         return model


# # # # from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

# # # # from stable_baselines3.common.callbacks import BaseCallback
# # # # from stable_baselines3.common.vec_env import VecEnv
# # # # import torch as th
# # # # import numpy as np
# # # # from gymnasium import spaces
# # # # from torch.nn import functional as F

# # # # from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union
# # # # from pathlib import Path  # Add this import
# # # # from io import BufferedIOBase  # Add this import as well since it's used

# # # # from stable_baselines3.common.buffers import RolloutBuffer
# # # # from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
# # # # from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
# # # # from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
# # # # from stable_baselines3.common.utils import explained_variance, obs_as_tensor
# # # # from stable_baselines3.a2c import A2C

# # # # from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
# # # # from stable_baselines3.common.utils import get_schedule_fn  # For learning rate scheduling

# # # # from agents.pure_mpc import PureMPC_Agent

# # # # SelfA2C = TypeVar("SelfA2C", bound="A2C")


# # # # class A2C_MPC(A2C):
# # # #     """
# # # #     Advantage Actor Critic (A2C)

# # # #     Paper: https://arxiv.org/abs/1602.01783
# # # #     Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
# # # #     and Stable Baselines (https://github.com/hill-a/stable-baselines)

# # # #     Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

# # # #     :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
# # # #     :param env: The environment to learn from (if registered in Gym, can be str)
# # # #     :param learning_rate: The learning rate, it can be a function
# # # #         of the current progress remaining (from 1 to 0)
# # # #     :param n_steps: The number of steps to run for each environment per update
# # # #         (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
# # # #     :param gamma: Discount factor
# # # #     :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
# # # #         Equivalent to classic advantage when set to 1.
# # # #     :param ent_coef: Entropy coefficient for the loss calculation
# # # #     :param vf_coef: Value function coefficient for the loss calculation
# # # #     :param max_grad_norm: The maximum value for the gradient clipping
# # # #     :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
# # # #         of RMSProp update
# # # #     :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
# # # #     :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
# # # #         instead of action noise exploration (default: False)
# # # #     :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
# # # #         Default: -1 (only sample at the beginning of the rollout)
# # # #     :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
# # # #     :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
# # # #     :param normalize_advantage: Whether to normalize or not the advantage
# # # #     :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
# # # #         the reported success rate, mean episode length, and mean reward over
# # # #     :param tensorboard_log: the log location for tensorboard (if None, no logging)
# # # #     :param policy_kwargs: additional arguments to be passed to the policy on creation
# # # #     :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
# # # #         debug messages
# # # #     :param seed: Seed for the pseudo random generators
# # # #     :param device: Device (cpu, cuda, ...) on which the code should be run.
# # # #         Setting it to auto, the code will be run on the GPU if possible.
# # # #     :param _init_setup_model: Whether or not to build the network at the creation of the instance
# # # #     """

# # # #     policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
# # # #         "MlpPolicy": ActorCriticPolicy,
# # # #         "CnnPolicy": ActorCriticCnnPolicy,
# # # #         "MultiInputPolicy": MultiInputActorCriticPolicy,
# # # #     }

# # # #     def __init__(
# # # #         self,
# # # #         mpcrl_cfg: Dict,
# # # #         version: str, 
# # # #         pure_mpc_cfg: Dict,
# # # #         policy: Union[str, Type[ActorCriticPolicy]],
# # # #         env: Union[GymEnv, str],
# # # #         learning_rate: Union[float, Schedule] = 7e-4,
# # # #         n_steps: int = 2048,
# # # #         batch_size: int = 64,
# # # #         gamma: float = 0.99,
# # # #         gae_lambda: float = 1.0,
# # # #         ent_coef: float = 0.0,
# # # #         vf_coef: float = 0.5,
# # # #         max_grad_norm: float = 0.5,
# # # #         rms_prop_eps: float = 1e-5,
# # # #         use_rms_prop: bool = True,
# # # #         use_sde: bool = False,
# # # #         sde_sample_freq: int = -1,
# # # #         rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
# # # #         rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
# # # #         normalize_advantage: bool = False,
# # # #         stats_window_size: int = 100,
# # # #         tensorboard_log: Optional[str] = None,
# # # #         policy_kwargs: Optional[Dict[str, Any]] = None,
# # # #         verbose: int = 0,
# # # #         seed: Optional[int] = None,
# # # #         device: Union[th.device, str] = "auto",
# # # #         _init_setup_model: bool = True,
# # # #     ):
# # # #         super().__init__(
# # # #             policy=policy,
# # # #             env=env,
# # # #             learning_rate=learning_rate,
# # # #             n_steps=n_steps,
# # # #             gamma=gamma,
# # # #             gae_lambda=gae_lambda,
# # # #             ent_coef=ent_coef,
# # # #             vf_coef=vf_coef,
# # # #             max_grad_norm=max_grad_norm,
# # # #             rms_prop_eps=rms_prop_eps,
# # # #             use_rms_prop=use_rms_prop,
# # # #             use_sde=use_sde,
# # # #             sde_sample_freq=sde_sample_freq,
# # # #             rollout_buffer_class=rollout_buffer_class,
# # # #             rollout_buffer_kwargs=rollout_buffer_kwargs,
# # # #             normalize_advantage=normalize_advantage,
# # # #             stats_window_size=stats_window_size,
# # # #             tensorboard_log=tensorboard_log,
# # # #             policy_kwargs=policy_kwargs,
# # # #             verbose=verbose,
# # # #             device=device,
# # # #             seed=seed,
# # # #             _init_setup_model=False,
# # # #         )

# # # #         # Update optimizer inside the policy if we want to use RMSProp
# # # #         # (original implementation) rather than Adam
# # # #         if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
# # # #             self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
# # # #             self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

# # # #         # Save A2C specific parameters
# # # #         self.normalize_advantage = normalize_advantage

# # # #         if _init_setup_model:
# # # #             self._setup_model()
            
# # # #         # Initialize MPC agent after setup_model
# # # #         self.version = version
# # # #         self.pure_mpc_cfg = pure_mpc_cfg
# # # #         self.mpc_agent = PureMPC_Agent(
# # # #             env=self.env.envs[0],
# # # #             cfg=pure_mpc_cfg,
# # # #         )
    
# # # #     def _setup_model(self) -> None:
# # # #         super()._setup_model()  # Basic setup is sufficient

# # # #     def train(self) -> None:
# # # #         """
# # # #         Update policy using the currently gathered
# # # #         rollout buffer (one gradient step over whole data).
# # # #         """
# # # #         # Switch to train mode (this affects batch norm / dropout)
# # # #         self.policy.set_training_mode(True)

# # # #         # Update optimizer learning rate
# # # #         self._update_learning_rate(self.policy.optimizer)

# # # #         # This will only loop once (get all data in one go)
# # # #         for rollout_data in self.rollout_buffer.get(batch_size=None):
# # # #             actions = rollout_data.actions
# # # #             if isinstance(self.action_space, spaces.Discrete):
# # # #                 # Convert discrete action from float to long
# # # #                 actions = actions.long().flatten()

# # # #             values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
# # # #             values = values.flatten()

# # # #             # Normalize advantage (not present in the original implementation)
# # # #             advantages = rollout_data.advantages
# # # #             if self.normalize_advantage:
# # # #                 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# # # #             # Policy gradient loss
# # # #             policy_loss = -(advantages * log_prob).mean()

# # # #             # Value loss using the TD(gae_lambda) target
# # # #             value_loss = F.mse_loss(rollout_data.returns, values)

# # # #             # Entropy loss favor exploration
# # # #             if entropy is None:
# # # #                 # Approximate entropy when no analytical form
# # # #                 entropy_loss = -th.mean(-log_prob)
# # # #             else:
# # # #                 entropy_loss = -th.mean(entropy)

# # # #             loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

# # # #             # Optimization step
# # # #             self.policy.optimizer.zero_grad()
# # # #             loss.backward()

# # # #             # Clip grad norm
# # # #             th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
# # # #             self.policy.optimizer.step()

# # # #         explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

# # # #         self._n_updates += 1
# # # #         self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
# # # #         self.logger.record("train/explained_variance", explained_var)
# # # #         self.logger.record("train/entropy_loss", entropy_loss.item())
# # # #         self.logger.record("train/policy_loss", policy_loss.item())
# # # #         self.logger.record("train/value_loss", value_loss.item())
# # # #         if hasattr(self.policy, "log_std"):
# # # #             self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

# # # #     def learn(
# # # #         self: SelfA2C,
# # # #         total_timesteps: int,
# # # #         callback: MaybeCallback = None,
# # # #         log_interval: int = 100,
# # # #         tb_log_name: str = "A2C",
# # # #         reset_num_timesteps: bool = True,
# # # #         progress_bar: bool = False,
# # # #     ) -> SelfA2C:
# # # #         return super().learn(
# # # #             total_timesteps=total_timesteps,
# # # #             callback=callback,
# # # #             log_interval=log_interval,
# # # #             tb_log_name=tb_log_name,
# # # #             reset_num_timesteps=reset_num_timesteps,
# # # #             progress_bar=progress_bar,
# # # #         )

# # # #     def collect_rollouts(
# # # #         self,
# # # #         env: VecEnv,
# # # #         callback: BaseCallback,
# # # #         rollout_buffer: RolloutBuffer,
# # # #         n_rollout_steps: int,
# # # #     ) -> bool:
# # # #         """
# # # #         Collect experiences using the current policy and fill a ``RolloutBuffer``.
# # # #         The term rollout here refers to the model-free notion and should not
# # # #         be used with the concept of rollout used in model-based RL or planning.

# # # #         :param env: The training environment
# # # #         :param callback: Callback that will be called at each step
# # # #             (and at the beginning and end of the rollout)
# # # #         :param rollout_buffer: Buffer to fill with rollouts
# # # #         :param n_rollout_steps: Number of experiences to collect per environment
# # # #         :return: True if function returned with at least `n_rollout_steps`
# # # #             collected, False if callback terminated rollout prematurely.
# # # #         """
# # # #         assert self._last_obs is not None, "No previous observation was provided"
# # # #         # Switch to eval mode (this affects batch norm / dropout)
# # # #         self.policy.set_training_mode(False)

# # # #         n_steps = 0
# # # #         rollout_buffer.reset()
# # # #         # Sample new weights for the state dependent exploration
# # # #         if self.use_sde:
# # # #             self.policy.reset_noise(env.num_envs)

# # # #         callback.on_rollout_start()

# # # #         while n_steps < n_rollout_steps:
# # # #             if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
# # # #                 # Sample a new noise matrix
# # # #                 self.policy.reset_noise(env.num_envs)

# # # #             with th.no_grad():
# # # #                 # Convert to pytorch tensor or to TensorDict
# # # #                 obs_tensor = obs_as_tensor(self._last_obs, self.device)
# # # #                 actions, values, log_probs = self.policy(obs_tensor)
# # # #             actions = actions.cpu().numpy()

# # # #             # Rescale and perform action
# # # #             clipped_actions = actions

# # # #             if isinstance(self.action_space, spaces.Box):
# # # #                 if self.policy.squash_output:
# # # #                     # Unscale the actions to match env bounds
# # # #                     # if they were previously squashed (scaled in [-1, 1])
# # # #                     clipped_actions = self.policy.unscale_action(clipped_actions)
# # # #                 else:
# # # #                     # Otherwise, clip the actions to avoid out of bound error
# # # #                     # as we are sampling from an unbounded Gaussian distribution
# # # #                     clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

# # # #             # let mpc agent work!
# # # #             mpc_action = self.mpc_agent.predict(
# # # #                 obs=np.squeeze(self._last_obs, axis=0),
# # # #                 return_numpy=True,
# # # #                 weights_from_RL=weights_from_RL,
# # # #                 ref_speed=ref_speed,
# # # #             )
# # # #             # print(clipped_actions.shape)
# # # #             mpc_action = mpc_action.reshape((1,2))

# # # #             new_obs, rewards, dones, infos = env.step(mpc_action)

# # # #             self.num_timesteps += env.num_envs

# # # #             # Give access to local variables
# # # #             callback.update_locals(locals())
# # # #             if not callback.on_step():
# # # #                 return False

# # # #             self._update_info_buffer(infos, dones)
# # # #             n_steps += 1

# # # #             if isinstance(self.action_space, spaces.Discrete):
# # # #                 # Reshape in case of discrete action
# # # #                 actions = actions.reshape(-1, 1)

# # # #             # Handle timeout by bootstraping with value function
# # # #             # see GitHub issue #633
# # # #             for idx, done in enumerate(dones):
# # # #                 if (
# # # #                     done
# # # #                     and infos[idx].get("terminal_observation") is not None
# # # #                     and infos[idx].get("TimeLimit.truncated", False)
# # # #                 ):
# # # #                     terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
# # # #                     with th.no_grad():
# # # #                         terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
# # # #                     rewards[idx] += self.gamma * terminal_value

# # # #             rollout_buffer.add(
# # # #                 self._last_obs,  # type: ignore[arg-type]
# # # #                 actions,
# # # #                 rewards,
# # # #                 self._last_episode_starts,  # type: ignore[arg-type]
# # # #                 values,
# # # #                 log_probs,
# # # #             )
# # # #             self._last_obs = new_obs  # type: ignore[assignment]
# # # #             self._last_episode_starts = dones

# # # #         with th.no_grad():
# # # #             # Compute value for the last timestep
# # # #             values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

# # # #         rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

# # # #         callback.update_locals(locals())

# # # #         callback.on_rollout_end()

# # # #         return True
    
# # # #     # def save(
# # # #     #     self,
# # # #     #     path: Union[str, Path, BufferedIOBase],
# # # #     #     exclude: Optional[List[str]] = None,
# # # #     #     include: Optional[List[str]] = None,
# # # #     # ) -> None:
# # # #     #     super().save(path, exclude, include)
# # # #     #     self.mpc_agent.save(str(path) + "_mpc")
# # # #     @classmethod
# # # #     def load(
# # # #         cls,
# # # #         path: Union[str, Path, BufferedIOBase],
# # # #         env: Optional[GymEnv] = None,
# # # #         device: Union[th.device, str] = "auto",
# # # #         force_reset: bool = True,
# # # #         **kwargs,
# # # #     ) -> "A2C_MPC":
# # # #         """
# # # #         Load the model from a zip-file

# # # #         :param path: path to the file (or a buffer to read from)
# # # #         :param env: the new environment to run the loaded model on
# # # #         :param device: Device on which the code should run.
# # # #         :param force_reset: Force call to ``env.reset()`` before training
# # # #         :param kwargs: extra arguments to change the model when loading
# # # #         :return: new model instance with loaded parameters
# # # #         """
# # # #         data, params, pytorch_variables = load_from_zip_file(path, device=device)

# # # #         # Remove stored device information and replace with ours
# # # #         if "policy_kwargs" in data:
# # # #             if "device" in data["policy_kwargs"]:
# # # #                 del data["policy_kwargs"]["device"]

# # # #         if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
# # # #             raise ValueError(
# # # #                 f"The specified policy kwargs do not equal the stored policy kwargs."
# # # #                 f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
# # # #             )

# # # #         if env is not None:
# # # #             # Wrap first if needed
# # # #             env = cls._wrap_env(env, data["verbose"])
# # # #             # Check if given env is valid
# # # #             check_for_correct_spaces(env, data["observation_space"], data["action_space"])
# # # #             # Discard `_last_obs`, this will force the env to reset before training
# # # #             if force_reset and data is not None:
# # # #                 data["_last_obs"] = None
# # # #             # `n_envs` must be updated
# # # #             if data is not None:
# # # #                 data["n_envs"] = env.num_envs
# # # #         else:
# # # #             # Use stored env, if one exists
# # # #             if "env" in data:
# # # #                 env = data["env"]

# # # #         # noinspection PyArgumentList
# # # #         model = cls(
# # # #             mpcrl_cfg=data.get("mpcrl_cfg"),
# # # #             version=data.get("version"),
# # # #             pure_mpc_cfg=data.get("pure_mpc_cfg"),
# # # #             policy=data["policy_class"],
# # # #             env=env,
# # # #             learning_rate=data.get("learning_rate", 7e-4),  # A2C default
# # # #             n_steps=data.get("n_steps", 5),  # A2C default
# # # #             gamma=data.get("gamma", 0.99),
# # # #             gae_lambda=data.get("gae_lambda", 1.0),  # A2C default
# # # #             ent_coef=data.get("ent_coef", 0.0),
# # # #             vf_coef=data.get("vf_coef", 0.5),
# # # #             max_grad_norm=data.get("max_grad_norm", 0.5),
# # # #             rms_prop_eps=data.get("rms_prop_eps", 1e-5),  # A2C specific
# # # #             use_rms_prop=data.get("use_rms_prop", True),  # A2C specific
# # # #             use_sde=data.get("use_sde", False),
# # # #             sde_sample_freq=data.get("sde_sample_freq", -1),
# # # #             normalize_advantage=data.get("normalize_advantage", False),
# # # #             stats_window_size=data.get("stats_window_size", 100),
# # # #             tensorboard_log=data.get("tensorboard_log", None),
# # # #             policy_kwargs=data.get("policy_kwargs", None),
# # # #             verbose=data.get("verbose", 0),
# # # #             seed=data.get("seed", None),
# # # #             device=device,
# # # #             _init_setup_model=False,
# # # #         )

# # # #         # load parameters
# # # #         model.__dict__.update(data)
# # # #         model.__dict__.update(kwargs)
# # # #         model._setup_model()

# # # #         # put state_dicts back in place
# # # #         model.set_parameters(params, exact_match=True, device=device)

# # # #         # put other pytorch variables back in place
# # # #         if pytorch_variables is not None:
# # # #             for name in pytorch_variables:
# # # #                 recursive_setattr(model, name, pytorch_variables[name])

# # # #         # Sample gSDE exploration matrix, so it uses the right device
# # # #         if model.use_sde:
# # # #             model.policy.reset_noise()

# # # #         if force_reset and model.get_env() is not None:
# # # #             model.env.reset()

# # # #         return model