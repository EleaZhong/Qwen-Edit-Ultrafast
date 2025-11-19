import pandas as pd

from collections import OrderedDict
from PIL import Image
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig, Int4WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig, ModuleFqnToConfig, PerRow
from qwenimage.debug import ctimed, ftimed
from qwenimage.experiment import ExperimentConfig
from qwenimage.experiments.experiments_qwen import ExperimentRegistry, QwenBaseExperiment
from qwenimage.models.attention_processors import QwenDoubleStreamAttnProcessorFA3
from qwenimage.optimization import optimize_pipeline_



@ExperimentRegistry.register(name="qwen_activations")
class Qwen_Activations(QwenBaseExperiment):
    
    def run(self):
        output_save_dir = self.config.report_dir / f"{self.config.name}_outputs"
        output_save_dir.mkdir(parents=True, exist_ok=True)

        self.pipe.transformer.start_recording_activations()
        for i in range(self.config.iterations):
            inputs = self.pipe_inputs[i]
            with ctimed("run_once"):
                output = self.run_once(**inputs)
            output.save(output_save_dir / f"{i:03d}.jpg")

    def report(self):
        act_report = self.pipe.transformer.end_recording_activations()
        print(act_report)
        act_means, act_maxs = act_report
        all_names = list(act_means.keys())
        print(all_names)
        data = []
        for name in all_names:
            actmean = act_means[name]
            actmax = act_maxs[name]
            data.append({
                'name': name,
                'mean': actmean,
                'max': actmax,
            })

        
        df = pd.DataFrame(data)
        df.to_csv(self.config.report_dir/f"transformer_activations.csv")
        return df, act_report
    
