# %%
import gym
import numpy as np
import torch as th
import typer
import kitchen
from GCP_utils.utils import (color_idx2color_pair, color_list,
                             total_orientation_list, total_template_list)
from stable_baselines3 import LangGCPPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from tqdm import tqdm

from talar.stable_baselines3.common.utils import obs_as_tensor
from talar.stable_baselines3.common.vec_env.base_vec_env import VecEnv
from typing import Optional

import gym


class TimeLimit(gym.Wrapper):
    """This wrapper will issue a `truncated` signal if a maximum number of timesteps is exceeded.

    If a truncation is not defined inside the environment itself, this is the only place that the truncation signal is issued.
    Critically, this is different from the `terminated` signal that originates from the underlying environment as part of the MDP.

    Example:
       >>> from gym.envs.classic_control import CartPoleEnv
       >>> from gym.wrappers import TimeLimit
       >>> env = CartPoleEnv()
       >>> env = TimeLimit(env, max_episode_steps=1000)
    """

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: Optional[int] = None,
    ):
        """Initializes the :class:`TimeLimit` wrapper with an environment and the number of steps after which truncation will occur.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps: An optional max episode steps (if ``Ǹone``, ``env.spec.max_episode_steps`` is used)
        """
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, done, info)`` with `truncated=True`
            if the number of steps elapsed >= max episode steps

        """
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            obs = self.env.reset()
            done = True
            info["TimeLimit.truncated"] = True
            

        return observation, reward, done, info

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            **kwargs: The kwargs to reset the environment with

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

color_arr = np.array(color_list)

from GCP_utils.language_description_for_kitchen import \
    onehot_idx_to_description


class Nop(object):
    def nop(*args, **kw):
        pass

    def __getattr__(agent, _):
        return agent.nop




def make_env():
    def _thunk():
        # env = LangGCPEnv(maximum_episode_steps=50, action_type='perfect', obs_type='order_invariant',
        #                direct_obs=True, use_subset_instruction=True,
        #                fail_dist=0.2,
        #                language_model_type='policy_ag',
        #                mode='train',
        #                )
        env = gym.make(
            'kitchen-low-v0',
            env_type='low',
            reward_type='prev_curr',
        )
        env.set_mode('train')
        env = Monitor(TimeLimit(env.unwrapped, max_episode_steps=200), None, allow_early_resets=True)
        
        return env

    return _thunk


def env_wrapper(num):
    envs = [
        make_env()
        for _ in range(num)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecNormalize(envs, norm_reward=True, norm_obs=False, training=False)

    return envs


def features2goal(language_features: th.Tensor, is_kitchen: bool =True) -> list:
    language_features = language_features.long()
    batch_size = language_features.shape[0]
    language_goal_list = []
    for idx in range(batch_size):
        if is_kitchen:
            language_goal_idx = language_features[idx].item()
            language_goal = onehot_idx_to_description[language_goal_idx]
        else:
            template_idx = language_features[idx, 0].item()
            orientation_idx = language_features[idx, 1].item()
            color_idx = language_features[idx, 2].item()

            template = total_template_list[template_idx]
            orientation = total_orientation_list[orientation_idx]
            color_idx_arr = np.array(color_idx2color_pair[color_idx])
            color_pair = color_arr[color_idx_arr]

            language_goal = template.format(color_pair[0], orientation, color_pair[1])
        
        language_goal_list.append(language_goal)
    
    return language_goal_list
# %%

def collect_rollouts(
    agent,
    env: VecEnv,
    buffer: dict,
    n_rollout_steps: int,
    stop_after_first: bool = False
) -> bool:
    """
    Collect experiences using the current policy and fill a ``RolloutBuffer``.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.

    :param env: The training environment
    :param callback: Callback that will be called at each step
        (and at the beginning and end of the rollout)
    :param rollout_buffer: Buffer to fill with rollouts
    :param n_rollout_steps: Number of experiences to collect per environment
    :return: True if function returned with at least `n_rollout_steps`
        collected, False if callback terminated rollout prematurely.
    """
    obs = env.reset()
    # Switch to eval mode (this affects batch norm / dropout)
    agent.policy.set_training_mode(False)

    for n_steps in tqdm(range(n_rollout_steps)):
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(obs, agent.device)
            # assert env.envs[0].language_goal == features2goal(obs_tensor[:, -1])[0], f'env_language_goal {env.envs[0].language_goal} != {features2goal(obs_tensor[:, -1])[0]}'
            actions, values, log_probs = agent.policy(obs_tensor)
        actions = actions.cpu().numpy()
        # print(features2goal(obs_tensor[:, -3:]))
        # Rescale and perform action
        clipped_actions = actions

        new_obs, rewards, dones, infos = env.step(clipped_actions)
        
        if stop_after_first and any(dones):
            break
            
        agent.num_timesteps += env.num_envs

        agent._update_info_buffer(infos, dones)

        if isinstance(agent.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        
        buffer['actions'].append(actions)
        buffer['obs'].append(obs[:, :-1])
        buffer['rewards'].append(rewards)
        buffer['dones'].append(dones)
        buffer['language_goal'].append(features2goal(obs_tensor[:, -1]))
        buffer['obs_goal'].append(obs[:, -1])
        
            
        # rollout_buffer.add(obs, actions, rewards, agent._last_episode_starts, values, log_probs)
        obs = new_obs
        
        agent._last_episode_starts = dones

    for key in buffer.keys():
        if key == 'language_goal':
            buffer[key] = np.array(buffer[key]).T.reshape(-1, 1)
        else:
            # reshape and concat
            np_buf = np.stack(buffer[key], axis=0)
            if len(np_buf.shape) == 2:
                # add a dimension
                np_buf = np_buf[:, :, None] # (n_steps, n_envs, 1)
                
            buffer[key] = np.vstack(np_buf.swapaxes(1, 0)) # (n_steps, n_envs, *shape)
        
        
    return buffer
    
# collect_rollouts(
#     agent,
#     env,
#     50,
#     stop_after_first=True  
# )
# %%
def main(NUM_ENVS: int = 1, N_STEPS: int = 20):
    env = env_wrapper(NUM_ENVS)
    agent = LangGCPPPO.load(
        '/home/ubuntu/talar/talar-openreview/talar_policy_seed0.zip'
    )

    buffer = {
        'actions': [],
        'obs': [],
        'rewards': [],
        'dones': [],
        'language_goal': [],
        'obs_goal': []        
    }
    # env.reset()
    # call once since the first episode is buggy for some reason
    buf = collect_rollouts(
        agent,
        env,
        buffer,
        N_STEPS,
    )
    from eztils import inspect
    inspect(buf)
    th.save(buf, 'kitchen_buf.pt')

if __name__ == '__main__': 
    # add typer
    typer.run(main)
    
# %%
# todo get success rate
# %%