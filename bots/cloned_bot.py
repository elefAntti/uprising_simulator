# GEMINI generated bot to load a neural network
# In a new bot file, e.g., `bots/cloned_bot.py`
import torch
import torch.nn as nn
from bots import register_bot
#from train_bc import BCNet # Reuse the network definition
from utils.obs_builder import EgoObsBuilder
from bots.utility_functions import get_base_coords, get_opponent_index

class BCNet(nn.Module):
    def __init__(self, obs_size=134, action_size=2):
        super(BCNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Tanh() # Actions are often scaled between -1 and 1. Tanh is perfect for this.
        )

    def forward(self, x):
        return self.network(x)

@register_bot
class ClonedBot:
    def __init__(self, agent_idx):
        self.idx = agent_idx
        self.builder = EgoObsBuilder() # Use the same obs builder
        self.model = BCNet()
        self.model.load_state_dict(torch.load("bc_policy.pth"))
        self.model.eval() # Set model to evaluation mode
        
    def _bases(self):
        base_own=get_base_coords(self.idx)
        base_opp=get_base_coords(get_opponent_index(self.idx))
        return base_own, base_opp


    def get_controls(self, bot_coords, green_coords, red_coords):
        # NOTE: You'll need to figure out base_own and base_opp
        # This is available in log_distill.py's wrapper
        base_own, base_opp = self._bases()

        obs_dict = self.builder.build(self.idx, bot_coords, red_coords, green_coords, base_own, base_opp)
        obs_tensor = torch.from_numpy(obs_dict["flat"]).float().unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            action_tensor = self.model(obs_tensor)

        actions = action_tensor.squeeze().cpu().numpy()
        return actions[0], actions[1]


@register_bot
class RLBot:
    def __init__(self, agent_idx):
        self.idx = agent_idx
        self.builder = EgoObsBuilder() # Use the same obs builder
        self.model = BCNet()
        self.model.load_state_dict(torch.load("rl_policy.pth"))
        self.model.eval() # Set model to evaluation mode
        
    def _bases(self):
        base_own=get_base_coords(self.idx)
        base_opp=get_base_coords(get_opponent_index(self.idx))
        return base_own, base_opp


    def get_controls(self, bot_coords, green_coords, red_coords):
        # NOTE: You'll need to figure out base_own and base_opp
        # This is available in log_distill.py's wrapper
        base_own, base_opp = self._bases()

        obs_dict = self.builder.build(self.idx, bot_coords, red_coords, green_coords, base_own, base_opp)
        obs_tensor = torch.from_numpy(obs_dict["flat"]).float().unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            action_tensor = self.model(obs_tensor)

        actions = action_tensor.squeeze().cpu().numpy()
        return actions[0], actions[1]
