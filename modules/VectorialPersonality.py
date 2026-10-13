"""
Vectorial Personality Module (4D)
Calculates the emotional physics of NPCs in real time using vector V = (X, Y, Z, W)
"""

from typing import Dict, List

# Temperament ids
NORMAL, SANGUINE, CHOLERIC, MELANCHOLIC, PHLEGMATIC = range(5)

# Value ranges per axis: X valence, Y initiative, Z stress, W inertia
_RANGES = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (0.0, 1.0))

DECAY_FACTOR = 0.1   # pull towards baseline per update (low inertia only)
LOW_INERTIA = 0.7    # W below this decays towards baseline
HIGH_INERTIA = 0.8   # W at/above this halves incoming change (persistence rule)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class VectorialPersonality:
    def __init__(self, npc_id: str, temperament: int = NORMAL):
        self.npc_id = npc_id
        # X = Affective Valence, Y = Initiative/Dominance, Z = Stress/Tension, W = Temporal Inertia
        self.baseline = [0.0, 0.0, 0.0, 0.5]
        self.vector = list(self.baseline)
        self.temperament = temperament

    def set_temperament(self, temperament_type: int) -> None:
        """Set the fixed temperament type"""
        self.temperament = temperament_type

    def compute_delta(self, target_vector: List[float]) -> List[float]:
        """Compute the delta between current and target vector"""
        return [self.vector[i] - target_vector[i] for i in range(4)]

    def apply_acceleration(self, delta_vector: List[float]) -> List[float]:
        """Return a NEW delta scaled by temperament and temporal inertia (input is untouched)."""
        d = list(delta_vector)
        if self.temperament == SANGUINE:        # optimistic: smoother transitions
            d[0] *= 0.8
        elif self.temperament == CHOLERIC:      # hostile: strong initiative, amplified negativity
            d[1] *= 1.5
            if d[0] < 0:
                d[0] *= 1.3
        elif self.temperament == MELANCHOLIC:   # withdrawn: stress-prone, dampened valence
            d[2] *= 1.4
            d[0] *= 0.9
        # PHLEGMATIC / NORMAL: no modifier

        if self.vector[3] >= HIGH_INERTIA:
            for i in range(3):
                d[i] *= 0.5
        return d

    def update_from_action(self, action_delta: List[float]) -> List[float]:
        """Apply a player action to the vector and return the delta that was applied.

        action_delta = [dX, dY, dZ, new W]. The input list is not modified.
        """
        applied = self.apply_acceleration(action_delta)

        for i in range(3):
            self.vector[i] += applied[i]

        # Low inertia: the emotion fades towards the baseline; high inertia: it sticks.
        if self.vector[3] < LOW_INERTIA:
            for i in range(3):
                self.vector[i] += (self.baseline[i] - self.vector[i]) * DECAY_FACTOR

        self.vector[3] = action_delta[3]
        self.vector = [_clamp(v, lo, hi) for v, (lo, hi) in zip(self.vector, _RANGES)]
        return applied

    def mood(self) -> str:
        """Coarse label for the current emotional state (used for the graph and LLM payload)."""
        valence, _, stress, _ = self.vector
        if stress >= 0.6:
            return "tense"
        if valence >= 0.25:
            return "friendly"
        if valence <= -0.25:
            return "hostile"
        return "neutral"

    def get_state(self) -> Dict:
        """Get current state as dictionary"""
        return {
            "npc_id": self.npc_id,
            "mood": self.mood(),
            "vector": {
                "affective_valence": round(self.vector[0], 3),
                "initiative_dominance": round(self.vector[1], 3),
                "stress_tension": round(self.vector[2], 3),
                "temporal_inertia": round(self.vector[3], 3),
            },
        }
