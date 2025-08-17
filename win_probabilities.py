import sys
from simulator import Simulator
from tqdm import tqdm
from tabulate import tabulate
import math
import argparse
import random

from bots import load_all_bots, get_bot_registry

load_all_bots()
bot_types = get_bot_registry()

def create_controllers(player_names):
    return [bot_types[player_names[i]](i) for i in range(4)]



def _wrap_controllers_with_sensor_noise(controllers, pos_sigma=0.02, angle_sigma_rad=math.radians(2.0), rng=None):
    """Monkey-patch each controller.get_controls to add Gaussian sensor noise.
    pos_sigma in same units as the simulator (meters). angle_sigma in radians.
    """
    rng = rng or random
    wrapped = []
    for ctrl in controllers:
        if not hasattr(ctrl, 'get_controls'):
            wrapped.append(ctrl)
            continue
        _orig = ctrl.get_controls  # bound method

        def noisy_get_controls(bot_coords, green_coords, red_coords, _o=_orig, _r=rng, _ps=pos_sigma, _as=angle_sigma_rad):
            def jxy(p):
                return (p[0] + _r.gauss(0.0, _ps), p[1] + _r.gauss(0.0, _ps))
            n_bot = [(jxy(p), ang + _r.gauss(0.0, _as)) for (p, ang) in bot_coords]
            n_green = [jxy(p) for p in green_coords]
            n_red = [jxy(p) for p in red_coords]
            return _o(n_bot, n_green, n_red)

        ctrl.get_controls = noisy_get_controls
        wrapped.append(ctrl)
    return wrapped
def simulate_game(player_names, pos_sigma=0.02, angle_sigma_deg=2.0, swap_sides_prob=0.5, rng=None, seed=None):
    controllers=create_controllers(player_names)
    rng = rng or random
    # Randomly swap sides (Team1 <-> Team2) per game with given probability
    flipped = False
    if rng.random() < swap_sides_prob:
        # swap [0,1] with [2,3]
        player_names = [player_names[2], player_names[3], player_names[0], player_names[1]]
        controllers = create_controllers(player_names)
        flipped = True
    
    # Wrap controllers with sensor noise
    controllers = _wrap_controllers_with_sensor_noise(
        controllers,
        pos_sigma=pos_sigma,
        angle_sigma_rad=math.radians(angle_sigma_deg),
        rng=rng
    )
    
    noise={
        # tweak relative sigmas (std dev of multiplicative noise around 1.0)
        "core_radius": 0.02,            # ~±5% cap in code
        "core_density": 0.15,
        "core_friction": 0.25,
        "core_restitution": 0.20,
        "core_linear_damping": 0.10,
        "core_angular_damping": 0.20,
        "robot_density": 0.10,
        "robot_friction": 0.25,
        "robot_ang_damp": 0.10,
        "robot_speed_scale": 0.05,      # set 0.0 to disable actuator randomness
    }
    simulator = Simulator()
    # Seed simulator (physics randomness) for reproducibility if seed provided
    sim_seed = seed if seed is not None else rng.randrange(1_000_000_000)
    simulator.init(controllers, True, noise=noise, seed=sim_seed)
    while not simulator.is_game_over():
        simulator.update()
    winner = simulator.get_winner()
    # If we flipped sides, map result back to original ordering for the caller:
    # 0->0 (draw), 1<->2 swap
    if flipped and winner in (1, 2):
        winner = 3 - winner
    return winner


def confidence_limit(p, n):
    return 1.96 * math.sqrt(p*(1.0 - p)/n)

def win_probabilities(player_names, n_games, pos_sigma=0.02, angle_sigma_deg=2.0, swap_sides_prob=0.5, base_seed=None):
    winners = [0, 0, 0]
    for i in tqdm(range(n_games)):
        rng = random.Random(base_seed + i) if base_seed is not None else random.Random()
        result = simulate_game(player_names, pos_sigma=pos_sigma, angle_sigma_deg=angle_sigma_deg, swap_sides_prob=swap_sides_prob, rng=rng, seed=(base_seed + i if base_seed is not None else None))
        winners[result] += 1
    probs = [x / float(n_games) for x in winners]
    return probs

parser = argparse.ArgumentParser(prog='win_probabilities', \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--games', type=int, default=200, help='How many games to simulate')

parser.add_argument('--sensor-pos-sigma', type=float, default=0.02, help='Std dev of position sensor noise (meters)')
parser.add_argument('--sensor-angle-sigma-deg', type=float, default=2.0, help='Std dev of angle sensor noise (degrees)')
parser.add_argument('--swap-sides-prob', type=float, default=0.5, help='Probability to swap sides (Team1 <-> Team2) each game')
parser.add_argument('--seed', type=int, default=None, help='Base RNG seed for reproducible experiments')

parser.add_argument('--tablefmt', type=str, default="fancy_grid", 
    help='How to format the resulttable eg github, fancy_grid or latex. \
        See https://pypi.org/project/tabulate/')
parser.add_argument('bots', nargs='*', default=[], help='''\
     0, 1, 2 or 4 bot names.
     0: Play all bots against each other
     1: Play all bots against the specified bot
     2: Play the selected bots against each other (2 in each team)
     4: Play the selected bots against each other
     ''')
args = parser.parse_args()
n_games = args.games

for name in args.bots:
    if name not in bot_types:
        print("'{}' isn't a registered bot class".format(name))
        parser.print_help()
        sys.exit(1)

if len(args.bots) == 0:
    bot_names = list(bot_types.keys())
    headers = ["Against"] + bot_names
    results = [[name] + [ "x" for _ in bot_names ] for name in bot_names]
    for i in range(len(bot_names)):
        for j in range(i):
            p1 = bot_names[i]
            p2 = bot_names[j]
            player_names = [p1, p1, p2, p2]
            print(p1 + " vs. " + p2)
            probs = win_probabilities(player_names, n_games, pos_sigma=args.sensor_pos_sigma, angle_sigma_deg=args.sensor_angle_sigma_deg, swap_sides_prob=args.swap_sides_prob, base_seed=args.seed)
            probstrings = ['{} \u00b1 {:.2f}'.format(prob, confidence_limit(prob, n_games))
                for prob in probs]
            results[i][j + 1] = probstrings[1]
            results[j][i + 1] = probstrings[2]
            print(tabulate(results, headers=headers, tablefmt = args.tablefmt))
elif len(args.bots) == 1:
    info = [['Against','Draw', 'Win', 'Lose']]
    for competitor in bot_types.keys():
        if competitor == args.bots[0]:
            continue
        print(args.bots[0] + " vs. " + competitor)
        player_names = [args.bots[0], args.bots[0], competitor, competitor]
        probs = win_probabilities(player_names, n_games, pos_sigma=args.sensor_pos_sigma, angle_sigma_deg=args.sensor_angle_sigma_deg, swap_sides_prob=args.swap_sides_prob, base_seed=args.seed)
        probstrings = ['{} \u00b1 {:.2f}'.format(prob, confidence_limit(prob, n_games))
            for prob in probs]
        info.append([competitor] + probstrings)
        print(tabulate(info, headers='firstrow', tablefmt = args.tablefmt))
elif len(args.bots) == 2:
    player_names = [args.bots[0], args.bots[0], args.bots[1], args.bots[1]]
    probs = win_probabilities(player_names, n_games, pos_sigma=args.sensor_pos_sigma, angle_sigma_deg=args.sensor_angle_sigma_deg, swap_sides_prob=args.swap_sides_prob, base_seed=args.seed)
    info = {'Result':['Draw', args.bots[0], args.bots[1]],
            'Prob.': probs,
            'Conf. Int.': [confidence_limit(probs[i], n_games) for i in range(3)]}

    print(tabulate(info, headers='keys', tablefmt = args.tablefmt))
elif len(args.bots) == 4:
    player_names = args.bots
    probs = win_probabilities(player_names, n_games, pos_sigma=args.sensor_pos_sigma, angle_sigma_deg=args.sensor_angle_sigma_deg, swap_sides_prob=args.swap_sides_prob, base_seed=args.seed)
    info = {'Result':['Draw', 'Team 1', 'Team 2'],
            'Prob.': probs,
            'Conf. Int.': [confidence_limit(probs[i], n_games) for i in range(3)]}

    print(tabulate(info, headers='keys', tablefmt = args.tablefmt))
else:
    parser.print_help()
    sys.exit(1)
