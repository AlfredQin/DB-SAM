import numpy as np

from super_gradients.common.registry.registry import register_callback
from super_gradients.training.utils.callbacks import Callback, PhaseContext, Phase
from super_gradients.training.utils.utils import unwrap_model, tensor_container_to_device
from super_gradients.training.utils.sg_trainer_utils import unpack_batch_items
from super_gradients.common.environment.ddp_utils import multi_process_safe


@register_callback()
class ComputeAverage3DDiceScore(Callback):
    def __init__(self, weights: dict = None):
        super().__init__()
        self.weights = weights or {}

    @multi_process_safe
    def on_test_loader_end(self, context: PhaseContext) -> None:
        """
        the format of task_name is "dataset_name:metric_name"
        """
        test_metrics_dict: dict = context.metrics_dict
        total_scores_3d, total_scores_2d = [], []
        for task_name, task_dice_score in test_metrics_dict.items():
            dataset_name, metric_name = task_name.split(":")
            if "2D" not in dataset_name and metric_name == "DICEScore":
                weight = self.weights.get(task_name, 1)
                total_scores_3d.append(float(weight * task_dice_score))
            elif "2D" in dataset_name and metric_name == "DICEScore":
                weight = self.weights.get(task_name, 1)
                total_scores_2d.append(float(weight * task_dice_score))
        average_score_3d = float(np.array(total_scores_3d).mean())
        context.metrics_dict.update({"Average_3D_DICE_Score": average_score_3d})
        average_score_2d = float(np.array(total_scores_2d).mean())
        context.metrics_dict.update({"Average_2D_DICE_Score": average_score_2d})



if __name__ == "__main__":
    callback = ComputeAverage3DDiceScore()
    context = PhaseContext()
    context.metrics_dict = {"task": 1}
    callback.on_test_loader_end(context)

    print()