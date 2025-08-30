# train_rl.py (Modified Version)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from robot_env import RobotGameEnv
from collections import OrderedDict
import torch
import torch.nn as nn
import argparse # New: Import for command-line arguments
import os       # New: Import for checking file paths

def load_bc_policy_into_sb3(model, bc_policy_path):
    """
    Helper function to load a BC policy .pth file into an SB3 model.
    """
    # 1) Inspect BC checkpoint
    bc_sd = torch.load(bc_policy_path, map_location="cpu")

    # 2) Remap BC -> SB3 policy keys
    remap = OrderedDict()
    try:
        remap["mlp_extractor.policy_net.0.weight"] = bc_sd["network.0.weight"]
        remap["mlp_extractor.policy_net.0.bias"]   = bc_sd["network.0.bias"]
        remap["mlp_extractor.policy_net.2.weight"] = bc_sd["network.2.weight"]
        remap["mlp_extractor.policy_net.2.bias"]   = bc_sd["network.2.bias"]
        remap["action_net.weight"] = bc_sd["network.4.weight"]
        remap["action_net.bias"]   = bc_sd["network.4.bias"]
    except KeyError as e:
        print(f"Error: A key was not found in the BC policy file. This can happen if the BCNet architecture has changed. Details: {e}")
        return

    # Initialize or copy log_std
    if "log_std" in bc_sd:
        remap["log_std"] = bc_sd["log_std"]
    else:
        with torch.no_grad():
            remap["log_std"] = torch.full_like(model.policy.log_std, -0.5)

    # 3) Load weights (critic weights will remain randomly initialized)
    missing, unexpected = model.policy.load_state_dict(remap, strict=False)
    print("Loaded BC policy into SB3 model.")
    print(f"  > Missing keys (expected, for value function): {missing}")
    print(f"  > Unexpected keys in source file: {unexpected}")


if __name__ == "__main__":
    # --- New: Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train or continue training an RL agent for the robot game.")
    parser.add_argument("--input-bc-policy", type=str, default=None,
                        help="Path to a pre-trained BC policy (.pth) to start training from.")
    parser.add_argument("--continue-rl-run", type=str, default=None,
                        help="Path to a previous RL run (.zip) to continue training.")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Number of timesteps to train for.")
    args = parser.parse_args()

    # --- Setup Environment ---
    env = make_vec_env(RobotGameEnv, n_envs=4)
    model = None

    # --- New: Conditional Model Loading ---
    if args.continue_rl_run:
        if os.path.exists(args.continue_rl_run):
            print(f"Loading and continuing RL training from {args.continue_rl_run}")
            model = PPO.load(args.continue_rl_run, env=env)
        else:
            print(f"Error: RL model file not found at {args.continue_rl_run}. Exiting.")
            exit()

    elif args.input_bc_policy:
        if os.path.exists(args.input_bc_policy):
            print(f"Starting new RL run, bootstrapping from BC policy at {args.input_bc_policy}")
            bc_sd = torch.load(args.input_bc_policy, map_location="cpu")
            h1 = bc_sd["network.0.weight"].shape[0]
            h2 = bc_sd["network.2.weight"].shape[0]

            # --- Corrected: Use ReLU for hidden layers ---
            activation = nn.ReLU

            policy_kwargs = dict(
                activation_fn=activation,
                net_arch=dict(pi=[h1, h2], vf=[h1, h2])
            )
            model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_robot_tensorboard/")
            load_bc_policy_into_sb3(model, args.input_bc_policy)
        else:
            print(f"Error: BC policy file not found at {args.input_bc_policy}. Exiting.")
            exit()
    else:
        print("Starting new RL training run from scratch.")
        # Default architecture if starting fresh
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_robot_tensorboard/")

    # --- Train the model ---
    if model:
        print(f"Starting training for {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps)

        # --- Save the final RL model ---
        output_filename = "rl_policy_final.zip"
        if args.continue_rl_run:
            # Overwrite the file we loaded from to continue progress
            output_filename = args.continue_rl_run
        print(f"Training complete. Saving model to {output_filename}")
        model.save(output_filename)
    else:
        print("Model was not initialized. Nothing to train.")
