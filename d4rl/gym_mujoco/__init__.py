from gym.envs.registration import register


AGENTS = ['Hopper', 'HalfCheetah', 'Ant', 'Walker2d']
DATASETS = ['random', 'medium', 'expert',
            'medium-expert', 'medium-replay', 'full-replay']


def register_gym_mujoco_envs():
    base_module = 'offline.d4rl.gym_mujoco.gym_envs'
    for agent in AGENTS:
        for dataset in DATASETS:
            env_name = f'{agent.lower()}-{dataset}-v2'
            register(id=env_name,
                     entry_point=f'{base_module}:Offline{agent}Env',
                     max_episode_steps=1000,
                     kwargs={'dataset_key': env_name})


register_gym_mujoco_envs()
