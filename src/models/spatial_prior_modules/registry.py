from super_gradients.common.registry.registry import create_register_decorator
from super_gradients.common.factories.base_factory import BaseFactory

SPATIAL_PRIOR_MODULES = {}
register_spatial_prior_module = create_register_decorator(SPATIAL_PRIOR_MODULES)

class SpatialPriorModuleFactory(BaseFactory):
    def __init__(self):
        super().__init__(SPATIAL_PRIOR_MODULES)