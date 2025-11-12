
import modal

from qwenimage.experiment import ExperimentConfig
from qwenimage.experiments.experiments_qwen import QwenBaseExperiment


app = modal.App("gradio-demo")
app.image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install_from_requirements(os.path.abspath("./requirements.txt"))
    .add_local_python_source("qwenimage")
)

experiment = QwenBaseExperiment(ExperimentConfig(name="qwen-base"))

experiment.load()
experiment.optimize()
experiment.run()
experiment.report()
