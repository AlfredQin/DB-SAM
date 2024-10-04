
from super_gradients.common.registry.registry import create_register_decorator
from super_gradients.common.factories.base_factory import BaseFactory

PROMPT_ENCODERS = {}
register_prompt_encoder = create_register_decorator(registry=PROMPT_ENCODERS)

class PromptEncoderFactory(BaseFactory):
    def __init__(self):
        super().__init__(PROMPT_ENCODERS)