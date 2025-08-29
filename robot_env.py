
# GEMINI made gymnasium environment for learning with RL
# robot_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Import your existing simulator and observation builder
from simulator import Simulator, red_core_coords, green_core_coords # and other game data
from utils.obs_builder import EgoObsBuilder
from bots import load_all_bots, get_bot_registry
import random
import bots.param_alias as PA

load_all_bots()

PA.autoload("zoo/**/*.json")

bot_types = get_bot_registry()

def clip(x, up, down):
    return max(min(x,up),down)

# Optional utility helpers if available
try:
    from bots.utility_functions import get_base_coords, get_opponent_index
except Exception:
    get_base_coords = None
    def get_opponent_index(i): return i^2  # 0<->2, 1<->3 by default

ARENA_W, ARENA_H = 1.5, 1.5

class RobotGameEnv(gym.Env):
    def __init__(self, agent_idx=0, goal_reward=10.0, goal_radius=0.10):
        super(RobotGameEnv, self).__init__()
        self.agent_idx = agent_idx # The index of our RL agent (0, 1, 2, or 3)

        # Reward tuning
        self.goal_reward = float(goal_reward)
        self.red_reward = 5.0 # extra reward for scoring a red ball
        self.goal_radius = float(goal_radius)  # meters; used for disappearance-based goal detection

        # Define the action and observation spaces
        # Actions are two continuous values (left/right wheel speed) between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observations are the 134-dimensional vector from your builder
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(134,), dtype=np.float32)

        # Initialize simulator and obs builder
        self.sim = Simulator()
        self.obs_builder = EgoObsBuilder()

        # We'll keep track of the distance to the nearest green ball for reward calculation
        self.prev_dist_to_green = None

        # Goal/score tracking (for dense event rewards)
        self.prev_scores = [0,0] 
        self._prev_red_ids = set()
        self._prev_green_ids = set()

        # Team mapping: indices 0,1 -> Team A ; 2,3 -> Team B
        self.own_team = 0 #"A" if (self.agent_idx in (0,1)) else "B"
        self.opp_team = 1 #"B" if self.own_team == "A" else "A"

    # ---------- Helpers ----------
    def _bases(self):
        # Prefer game utility if available
        if get_base_coords is not None:
            own = get_base_coords(self.agent_idx)
            opp = get_base_coords(get_opponent_index(self.agent_idx))
            return tuple(own), tuple(opp)
        # Fallback: corners
        if self.agent_idx in (0,1):
            own = (0.0, 0.0); opp = (ARENA_W, ARENA_H)
        else:
            own = (ARENA_W, ARENA_H); opp = (0.0, 0.0)
        return own, opp

    @staticmethod
    def _id_of(p, q=0.02):
        # Quantize to centimeters-ish to create stable ids for balls
        return (round(float(p[0]), 2), round(float(p[1]), 2))

    @staticmethod
    def _dist(a, b):
        ax, ay = a; bx, by = b
        return float(np.hypot(ax-bx, ay-by))

    def _read_scores(self):
        s = self.sim
        return s.scores[0], s.scores[1] 

    def _update_ball_id_sets(self):
        # Track current ids
        reds = [tuple(c.position) for c in getattr(self.sim, "red_cores", [])]
        greens = [tuple(c.position) for c in getattr(self.sim, "green_cores", [])]
        cur_red_ids = {self._id_of(p) for p in reds}
        cur_green_ids = {self._id_of(p) for p in greens}
        return cur_red_ids, cur_green_ids, reds, greens

    def _goal_events_since_last(self, prev_red_ids, prev_green_ids, reds, greens):
        """
        Fallback goal detector when Simulator doesn't expose score counters.
        If a ball disappears and its *last seen* location was inside a goal disc,
        we count it as a goal for the appropriate team.

        Rules of the game (as used elsewhere in this project):
          - Scoring for our team occurs when:
              * a RED enters the OPPONENT goal, or
              * a GREEN enters OUR goal.
          - Opponent scores conversely.
        """
        own_base, opp_base = self._bases()

        # Which ids vanished?
        cur_red_ids = {self._id_of(p) for p in reds}
        cur_green_ids = {self._id_of(p) for p in greens}
        vanished_red = prev_red_ids - cur_red_ids
        vanished_green = prev_green_ids - cur_green_ids

        own_delta = 0
        opp_delta = 0

        # To estimate where they vanished, we test proximity of *all current* coords won't help.
        # Instead, we assume that any vanished id close to a goal at last step would have had the same id;
        # but we only kept ids, not last positions. So we approximate by scanning previous ids' positions
        # at this step is not possible. For practical robustness we simply test: if an id vanishes and
        # EITHER goal currently has many balls near it, we attribute the vanish to that goal.
        # A better approach is to store last seen positions; we implement that here.
        # We'll keep a dict of last seen positions per id.
        if not hasattr(self, "_last_pos_by_id"):
            self._last_pos_by_id = {}
        # Update last pos for current ids
        for p in reds:
            self._last_pos_by_id[self._id_of(p)] = tuple(p)
        for p in greens:
            self._last_pos_by_id[self._id_of(p)] = tuple(p)

        # Evaluate vanished reds
        for rid in vanished_red:
            last = self._last_pos_by_id.get(rid, None)
            if last is None: continue
            if self._dist(last, opp_base) <= self.goal_radius:
                own_delta += 1   # our team got a red into opponent base
            elif self._dist(last, own_base) <= self.goal_radius:
                opp_delta += 1   # we accidentally scored red into our own base (opponent benefit)
            # Forget this id
            self._last_pos_by_id.pop(rid, None)

        # Evaluate vanished greens
        for gid in vanished_green:
            last = self._last_pos_by_id.get(gid, None)
            if last is None: continue
            if self._dist(last, own_base) <= self.goal_radius:
                own_delta += 1   # we got a green into our base
            elif self._dist(last, opp_base) <= self.goal_radius:
                opp_delta += 1
            self._last_pos_by_id.pop(gid, None)

        return own_delta, opp_delta

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Setup controllers: one RL agent, three scripted bots
        partner_type="Dagger" #random.choice(["SimpleBot", "PotentialWinner", "FuzzyPusher", "Dagger", "RedSlammer"])
        player_names = 4*[partner_type]#["SimpleBot", "SimpleBot", "SimpleBot", "SimpleBot"]
        controllers = [bot_types[player_names[i]](i) for i in range(4)]
        controllers[self.agent_idx] = self # The RL agent controls itself

        # Randomize starting positions for robust learning
        self.sim.init(controllers, randomize=True)

        # Reset observation & reward-related state
        obs = self._get_observation()

        self.prev_dist_to_ball = None

        # Initialize score/baseline
        scores = self._read_scores()
        if scores is not None:
            self.prev_scores = list(scores)
        self.reds = self.sim.red_core_counts
        # Initialize last id sets
        self._prev_red_ids, self._prev_green_ids, _, _ = self._update_ball_id_sets()
        self._last_pos_by_id = {}

        return obs, {}

    def step(self, action):
        # Provide action to simulator
        self.next_action = action
        self.sim.update()

        # Get new state
        obs = self._get_observation()
        terminated = bool(self.sim.is_game_over()) if hasattr(self.sim, "is_game_over") else False

        # --- Reward calculation ---
        reward = -0.001  # small time penalty

        # 1) Dense shaping toward greens (optional)
        balls = self.sim.get_green_cores() + self.sim.get_red_cores() 
        agent_pos = tuple(getattr(self.sim.robots[self.agent_idx], "position", (0.0,0.0)))
        dist_to_ball = None
        if balls:
            dist_to_ball = float(min(np.linalg.norm(np.array(agent_pos) - np.array(p)) for p in balls))
        if self.prev_dist_to_ball is not None and dist_to_ball is not None:
            reward += 0.1 * clip(self.prev_dist_to_ball - dist_to_ball, -0.1, 0.1)  # getting closer => positive
        self.prev_dist_to_ball = dist_to_ball

        # 2) Goal events: primary sparse reward
        delta_own = 0
        delta_opp = 0

        scores = self._read_scores()
        if scores is not None:
            # Score-based delta if Simulator exposes counters
            prev = self.prev_scores
            # Map team of agent to score keys
            own_key = self.own_team
            opp_key = self.opp_team
            delta_own = max(0, int(scores[own_key]) - int(prev[own_key]))
            delta_opp = max(0, int(scores[opp_key]) - int(prev[opp_key]))
            self.prev_scores = list(scores)

        # Apply event rewards
        reward += self.goal_reward * (delta_own - delta_opp)

        reds = self.sim.red_core_counts
        # Map team of agent to score keys
        own_key = self.own_team
        opp_key = self.opp_team
        delta_own = max(0, int(reds[own_key]) - int(self.reds[own_key]))
        delta_opp = max(0, int(reds[opp_key]) - int(self.reds[opp_key]))
        self.reds = list(reds)

        reward += self.red_reward * (delta_opp - delta_own)

        # And at the end of the step method, before returning:
        if terminated:
            winner = self.sim.get_winner()
            agent_team_id = 1 if self.agent_idx in (0, 1) else 2
            if winner == agent_team_id:
                reward += 50.0 # Large reward for winning
            elif winner != 0: # If there is a winner and it's not us
                reward -= 50.0 # Large penalty for losing

        # No time-limit truncation here
        truncated = False

        info =  {"delta_own_goals": int(delta_own), "delta_opp_goals": int(delta_opp)}
        return obs, float(reward), terminated, truncated, info

    # Called by simulator to get RL agent's controls
    def get_controls(self, bot_coords, green_coords, red_coords):
        a = getattr(self, "next_action", None)
        if a is None:
            return 0.0, 0.0
        return float(a[0]), float(a[1])

    def _get_observation(self):
        # Use your existing observation builder
        bot_coords = [(tuple(b.position), b.angle) for b in self.sim.robots]
        red_coords = [tuple(c.position) for c in self.sim.red_cores]
        green_coords = [tuple(c.position) for c in self.sim.green_cores]

        base_own, base_opp = self._bases()

        obs_dict = self.obs_builder.build(self.agent_idx, bot_coords, red_coords, green_coords, base_own, base_opp)
        return obs_dict["flat"]
