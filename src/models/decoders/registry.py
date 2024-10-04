
from super_gradients.common.registry.registry import create_register_decorator
from super_gradients.common.factories.base_factory import BaseFactory

DECODERS = {}
register_decoder = create_register_decorator(registry=DECODERS)

class DecoderFactory(BaseFactory):
    def __init__(self):
        super().__init__(DECODERS)