from .base import Skill
import numpy as np

class PinchSkill(Skill):
    name = "pinch"

    def sample_params(self, n=16):
        return [
            {
                "force": np.random.uniform(1.0, 5.0),
                "distance": np.random.uniform(0.01, 0.04),
            }
            for _ in range(n)
        ]
