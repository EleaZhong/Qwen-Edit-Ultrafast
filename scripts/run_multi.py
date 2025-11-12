import argparse
import subprocess
import sys

from qwenimage.experiments.experiments_qwen import ExperimentRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    
    experiment_names = ExperimentRegistry.keys()
    print(f"{len(experiment_names)}x {experiment_names}")
    
    for name in experiment_names:
        print(name)
        
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--name", name,
            "--iterations", str(args.iterations),
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(result)


if __name__ == "__main__":
    main()

