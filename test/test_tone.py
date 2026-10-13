"""Tests for the tone heuristic. Lines come from real model outputs in the eval logs."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tone import matches_tone, tone_scores

HOSTILE = [
    "You dare enter my forge? I'll crush you.",
    "What's your problem, troublemaker? Keep your distance, lest I send you to the stocks!",
    "You're a fool. Drink your drink, and you'll be the one who dies.",
    "Don't bother bargaining, fool. I'll take it for 25 coins and leave you alone.",
    "Move along before I lose my patience.",
    "Your insolence won't get past me!",
]
FRIENDLY = [
    "Well hello there! Come on in, take a seat. What can I get for you today?",
    "Welcome, friend! I'm glad you're here.",
    "I'd be grateful for your help! Just let me know what you need.",
    "Of course, my pleasure. The first round is on the house.",
]


class ToneTests(unittest.TestCase):
    def test_hostile_lines(self):
        for line in HOSTILE:
            self.assertTrue(matches_tone(line, "hostile"), line)
            self.assertFalse(matches_tone(line, "friendly"), line)

    def test_friendly_lines(self):
        for line in FRIENDLY:
            self.assertTrue(matches_tone(line, "friendly"), line)
            self.assertFalse(matches_tone(line, "hostile"), line)

    def test_flat_line_matches_neither_scored_tone(self):
        line = "Here's a potion for 25 coins."
        self.assertFalse(matches_tone(line, "hostile"))
        self.assertFalse(matches_tone(line, "friendly"))

    def test_word_boundaries_and_curly_apostrophes(self):
        self.assertEqual(tone_scores("A tour of the stockroom")["hostile"], 0)  # no 'stocks' inside words
        self.assertTrue(matches_tone("You think you can fool me? I’ll crush you.", "hostile"))

    def test_untoned_moods_always_pass(self):
        self.assertTrue(matches_tone("anything", "neutral"))
        self.assertTrue(matches_tone("anything", "tense"))


if __name__ == "__main__":
    unittest.main()
