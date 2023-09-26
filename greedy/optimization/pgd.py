"""
Created on Thur May 5 13:56 2022

@author: Jinpp (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Projected gradient optimizer, with box constraints.
@modifications: to be added
"""


import torch
from functools import reduce

from .line_search import *
from .optimizer import Optimizer

class PGD(Optimizer):
    
    def __init__(self,
                 params,
                 domain,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn)
        super(PGD, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("PGD doesn't support per-parameter options "
                        "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self._domain = domain

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    def _directional_evaluate_projected(self, closure, x, t, d):
        self._add_grad(t, d)
        self._box_projection()
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad        

    def _box_projection(self):
        index = 0
        for p in self._params:
            if p < self._domain[index][0]:
                p.copy_(self._domain[index][0])
            elif p > self._domain[index][1]:
                p.copy_(self._domain[index][1])
            index += 1

    @torch.no_grad()
    def step(self, closure):

        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        # compute gradient and define optimial condition
        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of number of iterations
            n_iter += 1
            state['n_iter'] += 1

            # compute gradient descent direction
            d = flat_grad.neg()

            # set attempt step for the armijo function
            s = 1e-1

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size for strong wolfe option
            if state['n_iter'] == 1:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                if line_search_fn == "strong_wolfe":
                    x_init = self._clone_param()

                    ##
                    self._add_grad(s, d)
                    self._box_projection()
                    d = torch.zeros(len(x_init))
                    for i in range(len(x_init)):
                        d[i] = self._params[i] - x_init[i]
                    self._set_param(x_init)
                    ##

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd)

                    self._add_grad(t, d)
                    self._box_projection()

                elif line_search_fn == "armijo":
                    x_init = self._clone_param()
                    self._add_grad(s, d)
                    self._box_projection()
                    d = torch.zeros(len(x_init))
                    for i in range(len(x_init)):
                        d[i] = self._params[i] - x_init[i]
                    self._set_param(x_init)

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = armijo(
                        obj_func, x_init, d, loss, flat_grad)

                    self._add_grad(t, d)
                    self._box_projection()

                elif line_search_fn == "arc_armijo":
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate_projected(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = armijo(
                        obj_func, x_init, d, loss, flat_grad)

                    self._add_grad(t, d)
                    self._box_projection()

                else:
                    raise RuntimeError("the line search algorithm is not supported")

                
                opt_cond = flat_grad.abs().max() <= tolerance_grad

            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            # print(flat_grad.abs().max())
            
            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss

