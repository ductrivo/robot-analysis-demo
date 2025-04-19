from abc import ABC, abstractmethod
from typing import override

import numpy as np
from numpy.typing import NDArray


class ControllerABC(ABC):
    def __init__(self, setpoint: NDArray, t_step: float, u_name: str = ''):
        self._setpoint = setpoint
        self._t_step = t_step
        self._n_setpoint = setpoint.shape[0]
        self._u_name = u_name
        self._u: NDArray

    @property
    def u(self) -> NDArray:
        return self._u

    @property
    def u_name(self) -> str:
        return self._u_name

    @abstractmethod
    def make_step(self, y: NDArray):
        pass


class FeedforwardPID(ControllerABC):
    def __init__(
        self,
        setpoint: NDArray,
        t_step: float,
        feedforward_term: NDArray | None = None,
        kP: NDArray | None = None,
        kI: NDArray | None = None,
        kD: NDArray | None = None,
        eI_max: NDArray | None = None,
        u_max: NDArray | None = None,
        u_name: str = '',
    ):
        super().__init__(setpoint, t_step, u_name)
        self._feedforward = feedforward_term
        self._kP = kP
        self._kI = kI
        self._kD = kD
        self._eI = np.zeros(self._setpoint.shape)
        self._eD = np.zeros(self._setpoint.shape)
        self._eI_max = eI_max
        self._u_max = u_max
        self._error_prev = -setpoint

    @override
    def make_step(self, y: NDArray):
        if self._feedforward is not None:
            self._u = self._feedforward
        else:
            self._u = np.zeros(self._setpoint.shape)

        if self._kP is not None:
            error = self._setpoint - y
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

        return self._u


class ComputedTorque(FeedforwardPID):
    def __init__(
        self,
        setpoint,
        t_step,
        feedforward_term=None,
        kP=None,
        kI=None,
        kD=None,
        eI_max=None,
        u_max=None,
    ):
        super().__init__(
            setpoint,
            t_step,
            feedforward_term,
            kP,
            kI,
            kD,
            eI_max,
            u_max,
            u_name='torque',
        )
        self.M: NDArray

    def make_step(self, y):
        # Kp*theta_e + Ki*eI + Kd*dtheta_e
        pid_term = super().make_step(y)
