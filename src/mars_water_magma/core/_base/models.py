"""Base Structures for Models."""

from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Callable

import numpy as np


class BaseModel(ABC):

    """Base Class for the Models."""

    def __init__(
        self,
        *,
        magma_temperature: float,
        initial_temperature: float,
        kappa: float,
    ) -> None:
        self._solution = None
        self.magma_temperature = magma_temperature
        self.initial_temperature = initial_temperature
        self.kappa = kappa

    @property
    def solution(self) -> Callable:
        if self._solution is None:
            self.start()
        return self._solution

    @abstractproperty
    def summary(self) -> str:
        """Summary of the Model."""

    @abstractmethod
    def start(self) -> None:
        """Start the Modelling."""

    def compute(self, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute the output of the model.

        Parameters
        ----------
        z : np.ndarray
            Spatial Domain (1D).
        t : np.ndarray
            Temporal Domain.

        Returns
        -------
        np.ndarray
            Solution.
        """
        return self.solution(z, t)
