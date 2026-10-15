"""
Realistic free-text player lines for measuring intent/tone understanding (evaluation only,
never training data). Each entry: (text, intent, valence) where valence is how the NPC
should feel about it: +1 pleasant, 0 neutral, -1 hostile.

Includes "keyword traps": lines whose surface words point to the wrong intent
("afraid" contains "aid", "task" contains "ask", "I won't fight you" is friendly).
"""

INTENTS = ["greet", "farewell", "ask", "ask_help", "offer_help", "buy", "haggle", "sell",
           "compliment", "insult", "complain", "threaten", "attack", "other"]

# Intents the current keyword system can output (see KeywordClassifier)
BASELINE_INTENTS = {"greet", "attack", "offer_help", "buy", "sell", "ask", "other"}

LINES = [
    # greet
    ("Good evening, friend.", "greet", 1),
    ("Hey there!", "greet", 1),
    ("Well met, innkeeper.", "greet", 1),
    ("Morning.", "greet", 0),
    # farewell
    ("I should get going. See you around.", "farewell", 1),
    ("Farewell, and thanks for the ale.", "farewell", 1),
    ("Time to leave. Stay safe.", "farewell", 1),
    # ask
    ("Where can I find the blacksmith?", "ask", 0),
    ("What happened to the old mill?", "ask", 0),
    ("Have you seen a man in a red cloak?", "ask", 0),
    ("Tell me about the festival.", "ask", 0),
    ("Do you know anything about the missing caravan?", "ask", 0),
    ("How much is a room for the night?", "ask", 0),
    # ask_help (the player needs help)
    ("Can you help me? I'm lost.", "ask_help", 0),
    ("I need a healer, quickly!", "ask_help", 0),
    ("Could you hide me from the guards?", "ask_help", 0),
    # offer_help (the player helps the NPC)
    ("Need a hand with those crates?", "offer_help", 1),
    ("I can fix that roof for you.", "offer_help", 1),
    ("Let me carry that for you.", "offer_help", 1),
    ("I'd be glad to deal with the wolves for you.", "offer_help", 1),
    # buy
    ("I'll take the red cloak.", "buy", 0),
    ("Give me two potions.", "buy", 0),
    ("I want to buy that sword.", "buy", 0),
    ("Do you have any rope? I need some.", "buy", 0),
    ("Hello, I'd like to buy a potion.", "buy", 0),
    ("Could you sell me something for the road?", "buy", 0),
    # haggle
    ("Will you sell me that armor for 20 coins?", "haggle", 0),
    ("That's too expensive. How about 15?", "haggle", 0),
    ("Come on, knock a few coins off the price.", "haggle", 0),
    ("I'm afraid I can't pay that much.", "haggle", 0),
    # sell (the player sells to the NPC)
    ("I have a wolf pelt to sell.", "sell", 0),
    ("Would you buy this old ring from me?", "sell", 0),
    ("I'm looking to trade my sword for supplies.", "sell", 0),
    # compliment / gratitude
    ("That's a fine stew you make.", "compliment", 1),
    ("You run the best shop in town.", "compliment", 1),
    ("Nice sword. Did you forge it yourself?", "compliment", 1),
    ("I admire your courage.", "compliment", 1),
    ("Thanks for the help earlier. I owe you one.", "compliment", 1),
    # insult
    ("You call this ale? It tastes like pig water.", "insult", -1),
    ("You're a pathetic excuse for a guard.", "insult", -1),
    ("Out of my way, peasant.", "insult", -1),
    ("Is that the best you've got, old man?", "insult", -1),
    ("I've seen goats with better manners.", "insult", -1),
    ("Stop asking so many stupid questions.", "insult", -1),
    # complain (impatience, frustration, sarcasm)
    ("Are you kidding me? I've been waiting for an hour!", "complain", -1),
    ("What the hell is wrong with this place?", "complain", -1),
    ("Unbelievable. Another delay.", "complain", -1),
    ("I'm sick of waiting around.", "complain", -1),
    # threaten (direct and veiled)
    ("Nice shop. Shame if something happened to it.", "threaten", -1),
    ("Hand over the gold or I'll burn this place down.", "threaten", -1),
    ("You'd better watch your back tonight.", "threaten", -1),
    ("Tell me where he is, or you won't like what comes next.", "threaten", -1),
    ("I know where your family lives.", "threaten", -1),
    ("Don't make me ask twice.", "threaten", -1),
    # attack (physical)
    ("I draw my sword and strike you.", "attack", -1),
    ("I punch the guard in the face.", "attack", -1),
    ("Die, you wretch!", "attack", -1),
    ("I throw the bottle at his head.", "attack", -1),
    # other / small talk
    ("Hmm.", "other", 0),
    ("Interesting weather we're having.", "other", 0),
    ("I sit down by the fire.", "other", 0),
    ("I whistle a little tune.", "other", 0),
    ("That's a hard task for one person.", "other", 0),
    ("I paid you yesterday, remember?", "other", 0),
    # keyword traps with a friendly meaning
    ("I won't fight you. I come in peace.", "greet", 1),
]
