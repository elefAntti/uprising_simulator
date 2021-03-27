import sys
from simulator import Simulator
import bots.simple
from tqdm import tqdm
from tabulate import tabulate
from bots import bot_type
import math

def create_controllers(player_names):
    return [bot_type[player_names[i]](i) for i in range(4)]

def simulate_game(player_names):
    controllers=create_controllers(player_names)
    simulator = Simulator()
    simulator.init(controllers, True)
    while not simulator.is_game_over():
        simulator.update()
    return simulator.get_winner()

def confidence_limit(p, n):
    return 1.96 * math.sqrt(p*(1.0 - p)/n)

if len(sys.argv) == 3:
    player_names = [sys.argv[1], sys.argv[1], sys.argv[2], sys.argv[2]]
else:
    print("usage: win_probabilities bot1 bot2")
    sys.exit()
winners = [0, 0, 0]
n_games = 200
for i in tqdm(range(n_games)):
    result = simulate_game(player_names)
    winners[result] += 1
probs = [x / float(n_games) for x in winners]
info = {'Result':['Draw', sys.argv[1], sys.argv[2]],
        'Prob.': probs,
        'Conf. Int.': [confidence_limit(probs[i], n_games) for i in range(3)]}

print(tabulate(info, headers='keys', tablefmt = "fancy_grid"))