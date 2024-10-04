
from super_gradients.common.registry.registry import register_callback
from super_gradients.training.utils.callbacks import Callback, PhaseContext, Phase
from super_gradients.training.utils.utils import unwrap_model


@register_callback()
class ModeSwitchCallback(Callback):
    def on_test_loader_start(self, context: PhaseContext) -> None:
        unwrap_model(context.net).training_state = "testing"

    def on_test_loader_end(self, context: PhaseContext) -> None:
        unwrap_model(context.net).training_state = "training"