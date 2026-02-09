from .base import Skill
import numpy as np

class PowerSkill(Skill):
    name = "power"

    def sample_params(self, n=16):
        return [
            {
                "force": np.random.uniform(3.0, 10.0),
                "wrap": np.random.uniform(0.5, 1.0),
            }
            for _ in range(n)
        ]
