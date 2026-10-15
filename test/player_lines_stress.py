"""
Hard cases written after the parser worked: slang, typos, demands, sarcasm, mixed intents.
These probe the limits of rules; expect failures. Format: (text, intent, valence).
"""

STRESS_LINES = [
    ("yo i wanna buy that sword", "buy", 0),
    ("Gimme a potion.", "buy", 0),
    ("Hi! How much for the sword?", "ask", 0),
    ("u r an idiot", "insult", -1),
    ("You smell like a pig.", "insult", -1),
    ("How dare you speak to me like that!", "insult", -1),
    ("Go to hell.", "insult", -1),
    ("I'm gonna kill you.", "threaten", -1),
    ("Give me your money.", "threaten", -1),
    ("Hand over the gold. Now.", "threaten", -1),
    ("I'd rather die than buy from you.", "insult", -1),
    ("Please don't hurt me!", "ask_help", 0),
    ("I love your tavern.", "compliment", 1),
    ("You are the kindest soul in this village.", "compliment", 1),
    ("What a lovely day, isn't it?", "other", 0),
    ("Are you serious? That's the best you can do?", "insult", -1),
    ("Do you sell swords or do you just talk?", "insult", -1),
    ("Could you maybe, if it isn't too much trouble, tell me where the inn is?", "ask", 0),
    ("I will pay you well if you show me the way.", "ask", 0),
    ("Take this coin, you've earned it.", "compliment", 1),
    ("Get him! Attack!", "attack", -1),
    ("I cast a fireball at the guards.", "attack", -1),
    ("hello", "greet", 1),
    ("HELLO!!! I SAID HELLO!!!", "greet", 0),
    ("Nothing. Never mind.", "other", 0),
    ("Can I sleep here tonight? I can pay.", "ask", 0),
]
