
import argparse

from qwenimage.debug import clear_cuda_memory, print_gpu_memory
from qwenimage.experiment import ExperimentConfig
from qwenimage.experiments.experiments_qwen import PipeInputs, Qwen_AoT, QwenBaseExperiment, ExperimentRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
        
    name = args.name
    
    pipe_inputs = PipeInputs()
    experiment = ExperimentRegistry.get(name)(
        config=ExperimentConfig(
            name=name,
            iterations=args.iterations,
        ), 
        pipe_inputs=pipe_inputs,
    )
    experiment.load()
    experiment.optimize()
    experiment.run()
    experiment.report()


if __name__ == "__main__":
    main()

