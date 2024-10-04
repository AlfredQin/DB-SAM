
from super_gradients.common.registry.registry import create_register_decorator
from super_gradients.common.factories.base_factory import BaseFactory

TRANSFORMERS = {}
register_transformer = create_register_decorator(registry=TRANSFORMERS)


class TransformerFactory(BaseFactory):
    def __init__(self):
        super().__init__(TRANSFORMERS)