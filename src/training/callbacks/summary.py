
from copy import deepcopy
from torchinfo import summary
from super_gradients.common.registry.registry import register_callback
from super_gradients.training.utils.callbacks import Callback, PhaseContext, Phase
from super_gradients.training.utils.utils import unwrap_model, tensor_container_to_device
from super_gradients.training.utils.sg_trainer_utils import unpack_batch_items
from super_gradients.common.environment.ddp_utils import multi_process_safe


@register_callback()
class TensorBoardAddModelGraphCallback(Callback):
    @multi_process_safe
    def on_train_loader_start(self, context: PhaseContext) -> None:
        if context.epoch == 0:
            first_batch = next(iter(context.train_loader))
            first_batch = tensor_container_to_device(first_batch, device=context.device, non_blocking=True)
            inputs, _, _ = unpack_batch_items(first_batch)
            model = deepcopy(unwrap_model(context.net))
            context.sg_logger.add_model_graph(model=model, dummy_input=inputs, tag=None)
        else:
            pass


@register_callback()
class TorchinfoSummaryCallback(Callback):
    @multi_process_safe
    def on_train_loader_start(self, context: PhaseContext) -> None:
        if context.epoch == 0:
            first_batch = next(iter(context.train_loader))
            first_batch = tensor_container_to_device(first_batch, device=context.device, non_blocking=True)
            inputs, _, _ = unpack_batch_items(first_batch)
            model = deepcopy(unwrap_model(context.net))
            summary(
                model,
                    input_data=[inputs],
                depth=8,
                mode="train",
                    col_names=["input_size", "output_size",
                    "num_params",
                    "params_percent",
                    "kernel_size",
                    "mult_adds",
                    "trainable",]
            )
        else:
            pass