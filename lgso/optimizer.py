from abc import ABC, abstractmethod
import numpy as np
from base_model import BaseConditionalGenerationOracle
from numpy.linalg import LinAlgError
from pyro import distributions as dist
from torch import optim
from torch import nn
from collections import defaultdict
import copy
import scipy
import matplotlib.pyplot as plt
import torch
import time
import sys
import copy
from scipy.stats import chi2

SUCCESS = 'success'
ITER_ESCEEDED = 'iterations_exceeded'
COMP_ERROR = 'computational_error'
SPHERE = True


class BaseOptimizer(ABC):
    """
    Base class for optimization of some function with logging
    functionality spread by all classes
    """
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 x_step: float = np.inf,  # step_data_gen
                 correct: bool = True,
                 tolerance: torch.Tensor = torch.tensor(1e-4),
                 trace: bool = True,
                 num_repetitions: int = 1000,
                 max_iters: int = 1000,
                 *args, **kwargs):
        self._oracle = oracle
        self._oracle.eval()
        self._history = defaultdict(list)
        self._x = x.clone().detach()
        self._x_init = copy.deepcopy(x)
        self._x_step = x_step
        self._tolerance = tolerance
        self._trace = trace
        self._max_iters = max_iters
        self._correct = correct
        self._num_repetitions = num_repetitions
        self._num_iter = 0.
        self._alpha_k = 0.
        self._previous_n_calls = 0

    def _update_history(self, init_time):
        self._history['time'].append(
            time.time() - init_time
        )
        self._history['func_evals'].append(
            self._oracle._n_calls - self._previous_n_calls
        )
        self._history['func'].append(
            self._oracle.func(self._x,
                              num_repetitions=self._num_repetitions).detach().cpu().numpy()
        )


        self._history['grad'].append(np.zeros_like(self._x.detach().cpu().numpy()))

        self._history['x'].append(
            self._x.detach().cpu().numpy()
        )
        self._history['alpha'].append(
            self._alpha_k
        )
        self._previous_n_calls = self._oracle._n_calls

    def optimize(self):
        """
        Run optimization procedure
        :return:
            torch.Tensor:
                x optim
            str:
                status_message
            defaultdict(list):
                optimization history
        """
        for i in range(self._max_iters):
            status = self._step()
            if status == COMP_ERROR:
                return self._x.detach().clone(), status, self._history
            elif status == SUCCESS:
                return self._x.detach().clone(), status, self._history
        return self._x.detach().clone(), ITER_ESCEEDED, self._history

    def update(self, oracle: BaseConditionalGenerationOracle, x: torch.Tensor, step=None):
        self._oracle = oracle
        self._x.data = x.data
        if step:
            self._x_step = step
        self._x_init = copy.deepcopy(x.detach().clone())
        self._history = defaultdict(list)

    @abstractmethod
    def _step(self):
        """
        Compute update of optimized parameter
        :return:
        """
        raise NotImplementedError('_step is not implemented.')

    def _post_step(self, init_time):
        """
        This function saves stats in history and forces
        :param init_time:
        :return:
        """
        if self._correct:
            if not SPHERE:
                self._x.data = torch.max(torch.min(self._x, self._x_init + self._x_step), self._x_init - self._x_step)
            else:
                # sphere cut
                x_corrected = self._x.data - self._x_init.data
                if x_corrected.norm() > self._x_step:
                    x_corrected = self._x_step * x_corrected / (x_corrected.norm())
                    x_corrected.data = x_corrected.data + self._x_init.data
                    self._x.data = x_corrected.data

        self._num_iter += 1
        if self._trace:
            self._update_history(init_time=init_time)

    def update_optimizer(self):
        pass


class TorchOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 torch_model: str = 'Adam',
                 optim_params: dict = {},
                 lr_algo: str = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._x.requires_grad_(True)
        self._lr = lr
        self._lr_algo = lr_algo
        self._alpha_k = self._lr
        self._torch_model = torch_model
        self._optim_params = optim_params
        self._base_optimizer = getattr(optim, self._torch_model)(
            params=[self._x], lr=lr, **self._optim_params
        )
        self._state_dict = copy.deepcopy(self._base_optimizer.state_dict())
        print(self._base_optimizer)

    def _step(self):
        init_time = time.time()
        self._base_optimizer.zero_grad()
        d_k = self._oracle.grad(self._x, num_repetitions=self._num_repetitions).detach()

        self._x.grad = d_k.detach().clone()
        self._state_dict = copy.deepcopy(self._base_optimizer.state_dict())
        self._base_optimizer.step()
        super()._post_step(init_time)
        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(self._x).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR

    def reverse_optimizer(self, **kwargs):
        self._base_optimizer.load_state_dict(self._state_dict)
        