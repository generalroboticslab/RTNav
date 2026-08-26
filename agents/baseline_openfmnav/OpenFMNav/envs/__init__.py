import torch

try:
    from .habitat import construct_envs, construct_envs21
except ImportError as exc:
    _HABITAT_IMPORT_ERROR = exc
    construct_envs = None
    construct_envs21 = None
else:
    _HABITAT_IMPORT_ERROR = None


def make_vec_envs(args):
    if construct_envs is None or construct_envs21 is None:
        raise ImportError(
            "OpenFMNav Habitat vector envs require the upstream Habitat-Lab API. "
            "The ROS wrapper imports mapping/planning utilities only and should "
            "not call make_vec_envs."
        ) from _HABITAT_IMPORT_ERROR
    if args.task_config == "tasks/objectnav_gibson.yaml":
        envs = construct_envs(args)
    else:
        envs = construct_envs21(args)
    envs = VecPyTorch(envs, args.device)
    return envs


# Adapted from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.device = device

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        obs, reward, done, info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def plan_act_and_preprocess(self, inputs):
        obs, reward, done, info = self.venv.plan_act_and_preprocess(inputs)
        obs = torch.from_numpy(obs).float().to(self.device)
        # reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def close(self):
        return self.venv.close()
