"""
Player lines written by Aaron (text.txt) as a real player might say them. The intent and
valence labels are CLAUDE'S reading, not Aaron's: review them. They were first measured
before any rule was changed for them, then used to fix whole classes of problems, so after
that this file is a dev set, not a clean test.

Placeholders like [NPC Name] are kept on purpose: the parser must cope with them.
Lines with no clear target ("Oh shit!") are labelled other.
"""

USER_LINES = [
    ("Hello there!", "greet", 1),
    ("Excuse me, [NPC Name], how much do you sell healing potions for?", "buy", 0),
    ("How is life treating you?", "ask", 0),
    ("I'm looking for a place to relax. I'm completely exhausted.", "ask", 0),
    ("Hey, did you hear what happened between the neighbor and her husband?", "ask", 0),
    ("Listen, dumbass, are you gonna help me or not?", "insult", -1),
    ("You're useless. I shouldn't have asked you for help.", "insult", -1),
    ("Who the hell do you think you are, trippin'?", "complain", -1),
    ("Excuse me, could you sign my petition?", "ask", 0),
    ("Damn it, [NPC_name], when are you going to help me get out of this?\"", "complain", -1),
    ("I wonder if they'll get pumpkins in next week for the holidays.", "other", 0),
    ("You said it, [NPC_Name], you said it.", "other", 0),
    ("Stop him RIGHT THERE!", "ask_help", 0),
    ("Good God, is anyone actually working here or what?", "complain", -1),
    ("Goddamn, that set is clean as hell!", "compliment", 1),
    ("The black goblin is very aggressive, I know.", "other", 0),
    ("God, look at the bastard move!", "other", 0),
    ("Oh, so NOW that affects me", "complain", -1),
    ("Wait, since when can you do that?", "ask", 0),
    ("We're doing good guys, we're doing good, we're doing good!", "other", 1),
    ("I'M TRYING!", "complain", -1),
    ("How is it possible that something so huge got so close to me without me noticing?", "ask", 0),
    ("Oh shit!", "other", 0),
    ("Hey stupid, get over here!", "insult", -1),
    ("Ever since everyone decided to start planting tulips too, sales haven't been going very well.", "other", 0),
]
