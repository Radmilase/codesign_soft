import json
from femtendon_simulator import FEMTendonSimulator  # Adjust based on actual import
from object_model import MustardBottle  # Adjust based on actual object model
import numpy as np

def run_experiment(num_trials=100):
    results = []
    simulator = FEMTendonSimulator(object_model=MustardBottle(), cpu_only=True)

    for trial in range(num_trials):
        result = simulator.run_simulation()
        results.append(result)

    return results

if __name__ == "__main__":
    results = run_experiment()
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)