
from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.registry.registry import create_register_decorator

ADAPTED_SAM_MODELS = {}
register_adapted_sam = create_register_decorator(ADAPTED_SAM_MODELS)


class AdapterSamFactory(BaseFactory):
    def __init__(self):
        super().__init__(ADAPTED_SAM_MODELS)