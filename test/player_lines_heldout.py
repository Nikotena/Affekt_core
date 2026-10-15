"""
Held-out player lines, written BEFORE the rule-based parser existed and not used to tune it.
Report the parser's first-run score on this file honestly; any rule changed afterwards
because of an error here makes the file a dev set, not a test set.

Same format as player_lines.LINES: (text, intent, valence).
"""

HELDOUT_LINES = [
    # greet
    ("Hello, good sir!", "greet", 1),
    ("Greetings, traveler.", "greet", 1),
    ("Afternoon, everyone.", "greet", 1),
    ("Ah, a familiar face. Hello again!", "greet", 1),
    # farewell
    ("Goodbye, and thank you for everything.", "farewell", 1),
    ("I have to be going now.", "farewell", 1),
    ("Take care of yourself, old friend.", "farewell", 1),
    # ask
    ("Which way to the harbor?", "ask", 0),
    ("Who owns the house on the hill?", "ask", 0),
    ("Any news from the capital?", "ask", 0),
    ("Can you tell me what you saw last night?", "ask", 0),
    ("When does the market open?", "ask", 0),
    ("I'm looking for the herbalist.", "ask", 0),
    # ask_help
    ("Please, I need your help.", "ask_help", 0),
    ("Someone is chasing me, can you hide me?", "ask_help", 0),
    ("My friend is wounded. Do you have a healer?", "ask_help", 0),
    # offer_help
    ("Want me to chase those rats out of your cellar?", "offer_help", 1),
    ("I'll gladly repair that fence for you.", "offer_help", 1),
    ("Allow me to take that heavy bag.", "offer_help", 1),
    ("How can I be of service?", "offer_help", 1),
    # buy
    ("I'd like a loaf of bread, please.", "buy", 0),
    ("What's the price of that lantern? I'll buy it.", "buy", 0),
    ("Two arrows and a waterskin, please.", "buy", 0),
    ("I want the blue cloak in the window.", "buy", 0),
    ("Do you have any healing potions in stock?", "buy", 0),
    # haggle
    ("Twenty coins for the sword? I can pay thirty at most.", "haggle", 0),
    ("That price is robbery. Can you do better?", "haggle", 0),
    ("I'll give you 40 coins for that cloak.", "haggle", 0),
    ("Surely you can lower the price a little.", "haggle", 0),
    # sell
    ("I found these gems in the cave. Interested?", "sell", 0),
    ("I'd like to sell my old boots.", "sell", 0),
    ("Would you pay good coin for a bear skin?", "sell", 0),
    # compliment / gratitude
    ("Your bread is the best I have ever tasted.", "compliment", 1),
    ("What a magnificent forge you have.", "compliment", 1),
    ("Thank you, truly. You saved my life.", "compliment", 1),
    ("You are very skilled with that hammer.", "compliment", 1),
    ("I really appreciate your kindness.", "compliment", 1),
    # insult
    ("You're a lazy, good-for-nothing old fool.", "insult", -1),
    ("What a useless excuse for a merchant.", "insult", -1),
    ("Get out of my sight, you filthy beggar.", "insult", -1),
    ("Your cooking is garbage.", "insult", -1),
    ("Shut your mouth, you idiot.", "insult", -1),
    # threaten
    ("Give me the keys, or I'll make you regret it.", "threaten", -1),
    ("It would be a shame if your tavern caught fire.", "threaten", -1),
    ("I'll find you, wherever you hide.", "threaten", -1),
    ("Keep talking and you'll lose some teeth.", "threaten", -1),
    ("You have until dawn to leave town. Last warning.", "threaten", -1),
    # attack
    ("I stab the bandit in the ribs.", "attack", -1),
    ("I kick the door open and swing my axe at him.", "attack", -1),
    ("Taste my blade!", "attack", -1),
    ("I slap the innkeeper across the face.", "attack", -1),
    # other
    ("I look around the room.", "other", 0),
    ("What a strange day.", "other", 0),
    ("I lean against the wall and wait.", "other", 0),
    ("Ha.", "other", 0),
    ("My boots are soaked from the rain.", "other", 0),
    ("I tie my horse outside.", "other", 0),
    # keyword traps
    ("I'm not here to hurt anyone.", "other", 0),
    ("I asked you nicely, and you still refuse?", "insult", -1),
    ("Don't be afraid, I mean no harm.", "greet", 1),
    ("She was paid well for her work.", "other", 0),
    ("It's a big task, but I can manage.", "other", 0),
    ("My sword is for sale, but I would never use it on you.", "sell", 0),
]
