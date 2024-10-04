
from super_gradients.common.registry.registry import create_register_decorator
from super_gradients.common.factories.base_factory import BaseFactory

ADAPTERS = {}
register_adapter = create_register_decorator(registry=ADAPTERS)

class AdapterFactory(BaseFactory):
    def __init__(self):
        super().__init__(ADAPTERS)