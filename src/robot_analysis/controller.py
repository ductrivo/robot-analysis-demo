from abc import ABC, abstractmethod


class Controller:
    def __init__(self, setpoint):
        self.setpoint = setpoint
