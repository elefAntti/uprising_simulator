import sys
from simulator import Simulator
import bots.simple
from tqdm import tqdm
from tabulate import tabulate
from bots import bot_type
import math
import argparse

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

def win_probabilities(player_names, n_games):
    winners = [0, 0, 0]
    for _ in tqdm(range(n_games)):
        result = simulate_game(player_names)
        winners[result] += 1
    probs = [x / float(n_games) for x in winners]
    return probs

parser = argparse.ArgumentParser(prog='win_probabilities', \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--games', type=int, default=200, help='How many games to simulate')
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
    if name not in bots.bot_type:
        print("'{}' isn't a registered bot class".format(name))
        parser.print_help()
        sys.exit(1)

if len(args.bots) == 0:
    bot_names = list(bots.bot_type.keys())
    headers = ["Against"] + bot_names
    results = [[name] + [ "x" for _ in bot_names ] for name in bot_names]
    for i in range(len(bot_names)):
        for j in range(i):
            p1 = bot_names[i]
            p2 = bot_names[j]
            player_names = [p1, p1, p2, p2]
            print(p1 + " vs. " + p2)
            probs = win_probabilities(player_names, n_games)
            probstrings = ['{} \u00b1 {:.2f}'.format(prob, confidence_limit(prob, n_games))
                for prob in probs]
            results[i][j + 1] = probstrings[1]
            results[j][i + 1] = probstrings[2]
            print(tabulate(results, headers=headers, tablefmt = args.tablefmt))
elif len(args.bots) == 1:
    info = [['Against','Draw', 'Win', 'Lose']]
    for competitor in bots.bot_type.keys():
        if competitor == args.bots[0]:
            continue
        print(args.bots[0] + " vs. " + competitor)
        player_names = [args.bots[0], args.bots[0], competitor, competitor]
        probs = win_probabilities(player_names, n_games)
        probstrings = ['{} \u00b1 {:.2f}'.format(prob, confidence_limit(prob, n_games))
            for prob in probs]
        info.append([competitor] + probstrings)
        print(tabulate(info, headers='firstrow', tablefmt = args.tablefmt))
elif len(args.bots) == 2:
    player_names = [args.bots[0], args.bots[0], args.bots[1], args.bots[1]]
    probs = win_probabilities(player_names, n_games)
    info = {'Result':['Draw', args.bots[0], args.bots[1]],
            'Prob.': probs,
            'Conf. Int.': [confidence_limit(probs[i], n_games) for i in range(3)]}

    print(tabulate(info, headers='keys', tablefmt = args.tablefmt))
elif len(args.bots) == 4:
    player_names = args.bots
    probs = win_probabilities(player_names, n_games)
    info = {'Result':['Draw', 'Team 1', 'Team 2'],
            'Prob.': probs,
            'Conf. Int.': [confidence_limit(probs[i], n_games) for i in range(3)]}

    print(tabulate(info, headers='keys', tablefmt = args.tablefmt))
else:
    parser.print_help()
    sys.exit(1)