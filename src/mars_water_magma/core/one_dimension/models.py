"""1D Model."""

import numpy as np
from scipy import sparse

from mars_water_magma.core._base.models import BaseModel


class SemiInfiniteSpaceModel(BaseModel):

    """Model for the case of a Semi-Infinite Space."""

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
        self._matrix = None
        super().__init__(
            magma_temperature=magma_temperature,
            initial_temperature=initial_temperature,
            kappa=kappa,
        )

    @property
    def matrix(self) -> sparse.lil_matrix:
        """Model's Matrix."""
        return self._matrix

    @property
    def summary(self) -> str:
        """Summary of the Model."""
        return (
            "-------------------\n"
            "\033[1mModel:\033[0m\n"
            "\tModel on a Semi-Infinite Space.\n"
            "-------------------\n"
            "\033[1mParameters:\033[0m\n"
            f"\tInitial temperature: {self.initial_temperature}°C\n"
            f"\tMagma temperature: {self.magma_temperature}°C\n"
            f"\tkappa : {self.kappa} W/m²\n"
            "------------------\n"
        )

    def create_matrix(self, *, z_dim: int, dt: float, dz: float) -> None:
        """Create the Matrix for the Model.

        Parameters
        ----------
        z_dim : int
            Space dimension.
        dt : float
            Time step.
        dz : float
            Space step.
        """
        matrix = sparse.lil_matrix((z_dim, z_dim), dtype=float)
        for i in range(1, z_dim - 1):
            matrix[i, i - 1] = 1
            matrix[i, i] = -2
            matrix[i, i + 1] = 1

        i_n = sparse.identity(z_dim, dtype="float")

        self._matrix = self.kappa * dt / (dz * dz) * matrix + i_n

    def start(self) -> None:
        """Start the Modelling."""

        def solution(z: np.ndarray, t: np.ndarray) -> np.ndarray:  # noqa: ARG001
            z_model = z.copy()
            for i in range(1, z.shape[0]):
                z0 = z_model[i - 1, :]
                z_model[i, :] = (self.matrix @ z0).reshape(1, -1)
            return z_model

        self._solution = solution
