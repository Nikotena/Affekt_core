"""
Vectorial Personality Module (4D)
Calculates the emotional physics of NPCs in real time using vector V = (X, Y, Z, W)
"""

import math
from typing import Dict, List, Tuple

class VectorialPersonality:
    def __init__(self, npc_id: str):
        self.npc_id = npc_id
        # Vector components: X = Affective Valence, Y = Initiative/Dominance, 
        # Z = Stress/Tension, W = Temporal Inertia
        self.vector = [0.0, 0.0, 0.0, 0.5]  # Default: neutral valence, no initiative, no stress, neutral inertia
        self.baseline = [0.0, 0.0, 0.0, 0.5]  # Baseline values
        self.temperament = 0  # 0=Normal, 1=Sanguine, 2=Choleric, 3=Melancholic, 4=Phlegmatic
    
    def set_temperament(self, temperament_type: int):
        """Set the fixed temperament type"""
        self.temperament = temperament_type
    
    def compute_delta(self, target_vector: List[float]) -> List[float]:
        """Compute the delta between current and target vector"""
        return [self.vector[i] - target_vector[i] for i in range(4)]
    
    def apply_acceleration(self, delta_vector: List[float], player_action_impact: float = 1.0):
        """Apply acceleration based on temperament and action impact"""
        # Apply temperament modifiers
        if self.temperament == 1:  # Sanguine - optimistic, social
            delta_vector[0] *= 0.8  # Smoother emotional transitions
        elif self.temperament == 2:  # Choleric - ambitious, hostile
            delta_vector[1] *= 1.5  # High initiative response
            if delta_vector[0] < 0:  # Negative emotion amplifies
                delta_vector[0] *= 1.3
        elif self.temperament == 3:  # Melancholic - thoughtful, withdrawn
            delta_vector[2] *= 1.4  # High stress response
            delta_vector[0] *= 0.9  # Emotional dampening
        elif self.temperament == 4:  # Phlegmatic - calm, steady
            # No significant modifier - steady state
            pass
        
        # Apply player action impact
        if self.vector[3] >= 0.8:  # High temporal inertia
            # Persistence rule: changes are permanent
            delta_vector[0] *= 0.5
            delta_vector[1] *= 0.5
            delta_vector[2] *= 0.5
        
        return delta_vector
    
    def update_from_action(self, action_delta: List[float]) -> List[float]:
        """Update vector from player action, return the actual delta"""
        total_delta = self.apply_acceleration(action_delta)
        
        # Apply decay (unless inertia locks changes)
        if self.vector[3] < 0.7:
            decay_factor = 0.1  # Decay to baseline
            for i in range(3):  # X, Y, Z decay
                target = self.baseline[i]
                diff = target - self.vector[i]
                self.vector[i] += diff * decay_factor * (1 - action_delta[3])
        else:
            # High inertia - changes persist more
            self.vector[0] += total_delta[0] * 0.6
            self.vector[1] += total_delta[1] * 0.6
            self.vector[2] += total_delta[2] * 0.6
        
        self.vector[3] = action_delta[3]  # Update temporal inertia
        
        # Clamp values to valid ranges
        self.vector[0] = max(-1.0, min(1.0, self.vector[0]))
        self.vector[1] = max(-1.0, min(1.0, self.vector[1]))
        self.vector[2] = max(0.0, min(1.0, self.vector[2]))
        self.vector[3] = max(0.0, min(1.0, self.vector[3]))
        
        return total_delta
    
    def get_state(self) -> Dict:
        """Get current state as dictionary"""
        return {
            "npc_id": self.npc_id,
            "vector": {
                "affective_valence": round(self.vector[0], 3),
                "initiative_dominance": round(self.vector[1], 3),
                "stress_tension": round(self.vector[2], 3),
                "temporal_inertia": round(self.vector[3], 3)
            }
        }