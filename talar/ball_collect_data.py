# %%
import gym
import numpy as np
import torch as th
from tqdm import tqdm
import typer
from envs.clevr_robot_env.env import LangGCPEnv
from GCP_utils.utils import (color_idx2color_pair, color_list,
                             total_orientation_list, total_template_list)
from stable_baselines3 import LangGCPPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from talar.stable_baselines3.common.utils import obs_as_tensor
from talar.stable_baselines3.common.vec_env.base_vec_env import VecEnv
from tensordict.tensordict import TensorDict as td
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
        env = LangGCPEnv(maximum_episode_steps=50, action_type='perfect', obs_type='order_invariant',
                       direct_obs=True, use_subset_instruction=True,
                       fail_dist=0.2,
                       language_model_type='policy_ag',
                       mode='train',
                       )

        env = Monitor(env, None, allow_early_resets=True)
        
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


def features2goal(language_features: th.Tensor, is_kitchen: bool =False) -> list:
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
            # assert env.envs[0].language_goal == features2goal(obs_tensor[:, -3:])[0], f'env_language_goal {env.envs[0].language_goal} != {features2goal(obs_tensor[:, -3:])[0]}'
            actions, values, log_probs = agent.policy(obs_tensor)
        actions = actions.cpu().numpy()
        # print(features2goal(obs_tensor[:, -3:]))
        # Rescale and perform action
        clipped_actions = actions

        new_obs, rewards, dones, infos = env.step(clipped_actions)
        agent.num_timesteps += env.num_envs

        agent._update_info_buffer(infos, dones)

        if isinstance(agent.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        buffer['actions'].append(actions)
        buffer['obs'].append(obs[:, :-3])
        buffer['rewards'].append(rewards)
        buffer['dones'].append(dones)
        buffer['language_goal'].append(features2goal(obs_tensor[:, -3:]))
        buffer['obs_goal'].append(obs[:, -3:])
        buffer['scene_graph'].append(np.array(list(i['scene_graph'] for i in infos)))
           
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
def main(NUM_ENVS: int = 2, N_STEPS: int = 20):
    env = env_wrapper(NUM_ENVS)
    agent = LangGCPPPO.load(
        '/home/ubuntu/talar/talar-openreview/talar/ball_model/langGCP_ppogcp_num_2_lr_0.0003_ns_512_bs_512_fd_0.2_lm_policy_ag_nobj_5_seed_0_hidden_128_output_14/model_50012.zip'
    )

    buffer = {
        'actions': [],
        'obs': [],
        'rewards': [],
        'dones': [],
        'language_goal': [],
        'obs_goal': [], 
        'scene_graph': []
    }
    # env.reset()
    # call once since the first episode is buggy for some reason
    buf = collect_rollouts(
        agent,
        env,
        buffer,
        N_STEPS,
    )
    # from eztils import inspect
    # inspect(buf)
    from eztils import datestr
    th.save(buf, f'buf_{datestr(full=False)}.pt')

if __name__ == '__main__': 
    # add typer
    typer.run(main)
    
# %%
# todo get success rate
# %%