
from super_gradients.common.registry.registry import create_register_decorator
from super_gradients.common.factories.base_factory import BaseFactory

Encoders = {}
register_encoder = create_register_decorator(registry=Encoders)

class EncoderFactory(BaseFactory):
    def __init__(self):
        super().__init__(Encoders)