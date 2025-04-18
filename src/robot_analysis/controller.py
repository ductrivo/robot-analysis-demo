from abc import ABC, abstractmethod
from typing import override

import numpy as np
from numpy.typing import NDArray


class ControllerABC(ABC):
    def __init__(self, setpoint: NDArray, t_step: float):
        self._setpoint = setpoint
        self._t_step = t_step
        self._n_setpoint = setpoint.shape[0]

    @abstractmethod
    def make_step(self, output: NDArray):
        pass


class PID(ControllerABC):
    def __init__(
        self,
        setpoint: NDArray,
        t_step: float,
        u_desired: NDArray,
        kP: NDArray | None = None,
        kI: NDArray | None = None,
        kD: NDArray | None = None,
        eI_max: NDArray | None = None,
        u_max: NDArray | None = None,
    ):
        super().__init__(setpoint, t_step)
        self._u_desired = u_desired
        self._kP = kP
        self._kI = kI
        self._kD = kD
        self._eI = np.zeros(self._setpoint.shape)
        self._eD = np.zeros(self._setpoint.shape)
        self._eI_max = eI_max
        self._u_max = u_max
        self._error_prev = -setpoint

    @override
    def make_step(self, output: NDArray):
        self._u = self._u_desired

        if self._kP is not None:
            error = self._setpoint - output
            self._u += self._kP * error

        if self._kI is not None:
            self._eI += error * self._t_step
            if self._eI_max is not None:
                self._eI = np.clip(self._eI, -self._eI_max, self._eI_max)
            self._u += self._kI * self._eI

        if self._kD is not None:
            self._eD = (error - self._error_prev) / self._t_step
            self._u += self._kD * self._eD

        if self._u_max is not None:
            self._u = np.clip(self._u, -self._u_max, self._u_max)


class VelocityPID(PID):
    def __init__(
        self,
        setpoint,
        t_step,
        u_desired,
        kP=None,
        kI=None,
        kD=None,
        eI_max=None,
        u_max=None,
    ):
        super().__init__(
            setpoint,
            t_step,
            u_desired,
            kP,
            kI,
            kD,
            eI_max,
            u_max,
        )

    @override
    def make_step(self, output):
        return super().make_step(output)
