import sys
from simulator import Simulator
import bots.simple
from tqdm import tqdm
from tabulate import tabulate
from bots import bot_type

def create_controllers(player_names):
    return [bot_type[player_names[i]](i) for i in range(4)]

def simulate_game(player_names):
    controllers=create_controllers(player_names)
    simulator = Simulator()
    simulator.init(controllers, True)
    while not simulator.is_game_over():
        simulator.update()
    return simulator.get_winner()


if len(sys.argv) == 3:
    player_names = [sys.argv[1], sys.argv[1], sys.argv[2], sys.argv[2]]
else:
    print("usage: win_probabilities bot1 bot2")
    sys.exit()
winners = [0, 0, 0]
n_games = 100
for i in tqdm(range(n_games)):
    result = simulate_game(player_names)
    winners[result] += 1

info = {'Result':['Draw', sys.argv[1], sys.argv[2]], 'Count': winners}

print(tabulate(info, headers='keys', tablefmt = "fancy_grid"))