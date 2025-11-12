


from qwenimage.experiment import ExperimentConfig
from qwenimage.experiments.experiments_qwen import Qwen_AoT, QwenBaseExperiment, ExperimentRegistry

# experiment = QwenBaseExperiment(ExperimentConfig(name="qwen-base"))
# experiment.load()
# experiment.optimize()
# experiment.run()
# experiment.report()

experiment = Qwen_AoT(ExperimentConfig(name="qwen-aot"))
experiment.load()
experiment.optimize()
experiment.run()
experiment.report()


