
from ..optimization import *

class Generator():
    
    def __init__(self,
                 parameters,
                 param_domain,
                 ):
        self.parameters = parameters
        self.param_domain = param_domain
        
    def _get_optimizer_dict(self):
        op_dict = {
                    "pgd": PGD(self.parameters,
                               self.param_domain,
                               lr=1e-2,
                               max_iter=20,
                               max_eval=None,
                               tolerance_grad=1e-08,
                               tolerance_change=1e-10,  # np.finfo(float).eps,
                               history_size=100,
                               line_search_fn="arc_armijo"),  # arc_armijo > strong_wolfe > armijo 
                    
                    "fista": FISTA(self.parameters,
                                   self.param_domain,
                                   lr=1e-2,
                                   max_iter=5,
                                   max_eval=None,
                                   tolerance_grad=1e-08,
                                   tolerance_change=1e-10,
                                   history_size=100,
                                   line_search_fn="arc_armijo"),  # arc_armijo 
                    
                    "lbfgs": LBFGS(self.parameters,
                                   lr=1,
                                   max_iter=20,
                                   max_eval=None,
                                   tolerance_grad=1e-08,
                                   tolerance_change=1e-10,
                                   history_size=100,
                                   line_search_fn="strong_wolfe"),  # only strong wolfe is supported 
                }
        return op_dict
    
    def get_optimizer(self, optimizer_type):
        op_dict = self._get_optimizer_dict()
        optimizer = op_dict.get(optimizer_type)
        if optimizer is None:
            raise RuntimeError("Such optimizer is not supported.")
        else:
            return optimizer
        