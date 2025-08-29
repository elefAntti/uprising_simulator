# train_rl.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from robot_env import RobotGameEnv
from collections import OrderedDict
import torch

# Path to your pre-trained model
BC_POLICY_PATH = "dagger_policy.pth"

# 1. Create the environment
# make_vec_env helps run multiple environments in parallel for faster training
env = make_vec_env(RobotGameEnv, n_envs=4)

# 2. Define the PPO model
#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_robot_tensorboard/")

# 3. *** Load the pre-trained weights from your BC policy ***
# We only load the policy network's weights, not the whole file.
#model.policy.load_state_dict(torch.load(BC_POLICY_PATH))

# 1) Inspect BC checkpoint to infer architecture
bc_sd = torch.load(BC_POLICY_PATH, map_location="cpu")
h1 = bc_sd["network.0.weight"].shape[0]   # e.g., 128
h2 = bc_sd["network.2.weight"].shape[0]   # e.g., 128

# If you know your BC used Tanh, set Tanh; otherwise ReLU
import torch.nn as nn
activation = nn.Tanh  # or nn.Tanh

# 2) Create SB3 model with matching policy MLP (make pi latent = h2)
policy_kwargs = dict(
    activation_fn=activation,
    net_arch=dict(pi=[h1, h2], vf=[h1, h2])  # vf can differ; keeping same is simplest
)

# IMPORTANT: create the env BEFORE this
# env = ...
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# 3) Remap BC -> SB3 policy keys
remap = OrderedDict()
remap["mlp_extractor.policy_net.0.weight"] = bc_sd["network.0.weight"]
remap["mlp_extractor.policy_net.0.bias"]   = bc_sd["network.0.bias"]
remap["mlp_extractor.policy_net.2.weight"] = bc_sd["network.2.weight"]
remap["mlp_extractor.policy_net.2.bias"]   = bc_sd["network.2.bias"]
remap["action_net.weight"] = bc_sd["network.4.weight"]
remap["action_net.bias"]   = bc_sd["network.4.bias"]

# Initialize or copy log_std
if "log_std" in bc_sd:
    remap["log_std"] = bc_sd["log_std"]
else:
    with torch.no_grad():
        if hasattr(model.policy, "log_std"):
            remap["log_std"] = torch.full_like(model.policy.log_std, -0.5)
        else:
            remap["log_std"] = torch.full((model.action_space.shape[0],), -0.5)

# 4) Load (critic weights will remain randomly initialized)
missing, unexpected = model.policy.load_state_dict(remap, strict=False)
print("Loaded BC → SB3. Missing:", missing, "Unexpected:", unexpected)


# Sanity checks (optional)
# Verify shapes match
assert model.policy.action_net.weight.shape == remap["action_net.weight"].shape
assert model.policy.action_net.bias.shape   == remap["action_net.bias"].shape


# 4. Train the model
# Start with a smaller number of timesteps to test, then increase.
model.learn(total_timesteps=100_000)

# 5. Save the final RL model
model.save("rl_policy")
