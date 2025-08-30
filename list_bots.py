from bots import load_all_bots, get_bot_registry
import bots.param_alias as PA

load_all_bots()

PA.autoload("zoo/**/*.json")

REG = get_bot_registry()

print("\n".join(sorted(REG.keys())))
