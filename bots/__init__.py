bot_type = {}

def register_bot(bot):
    bot_type[bot.__name__] = bot
    return bot

keyboard_listeners=[]
