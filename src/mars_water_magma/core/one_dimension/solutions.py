"""Benchmark Model for the 1D Diffusion Problem."""

import numpy as np
from scipy.special import erf

from mars_water_magma.core._base.models import BaseModel


class SemiInfiniteSpaceSolution(BaseModel):

    """Analytic Solution in the Case of a Semi-Finite Space."""

    def __init__(
        self,
        *,
        magma_temperature: float,
        initial_temperature: float,
        kappa: float,
    ) -> None:
        """Instantiate Model.

        Parameters
        ----------
        magma_temperature : float
            Temperature of the magma.
        initial_temperature : float
            Initial Temperature.
        kappa : float
            Kappa.
        """
        super().__init__(
            magma_temperature=magma_temperature,
            initial_temperature=initial_temperature,
            kappa=kappa,
        )

    @property
    def summary(self) -> str:
        """Summary of the Model."""
        return (
            "-------------------\n"
            "\033[1mModel:\033[0m\n"
            "\tAnalytic Solution on a Semi Finite Space.\n"
            "-------------------\n"
            "\033[1mParameters:\033[0m\n"
            f"\tInitial temperature: {self.initial_temperature}°C\n"
            f"\tMagma temperature: {self.magma_temperature}°C\n"
            f"\tkappa : {self.kappa} W/m²\n"
            "------------------\n"
        )

    def start(
        self,
    ) -> np.ndarray:
        """Start the modelling process.

        Returns
        -------
        np.ndarray
            output.
        """

        def solution(z: float, t: float) -> float:
            delta_t = self.initial_temperature - self.magma_temperature
            erf_temperature = erf(z / (2 * np.sqrt(self.kappa * t)))
            return self.magma_temperature + delta_t * erf_temperature

        self._solution = np.vectorize(solution)
