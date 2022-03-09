from abc import ABC, abstractmethod
from typing import Callable, Type

import numpy as np

from utils.type_aliases import is_scalar_64_tg


class Sequence(ABC):

    def __init__(self, *, decay_parameter: str,
                 target_iterations: int|None = None, target_value: float|None = None,
                 initial_value: float| None = None, constant_value: float|None = None,
                 n0: float|None = None): # pylint: disable=invalid-name


        self.decay_parameter: str = decay_parameter

        self.target_iterations: int|None = target_iterations
        self.target_value: float|None = target_value
        self.constant_value: float|None = constant_value
        self.initial_value: float|None = initial_value
        self.n0: float|None = n0 # pylint: disable=invalid-name


    @classmethod
    @abstractmethod
    def get_config_parameters(cls) -> tuple[str,...]:
        pass


    @abstractmethod
    def _get_current_value(self, **kwargs: int) -> float:
        pass


    def get_schedule(self) -> Callable[..., float]:
        return self._get_current_value


class Constant(Sequence):

    @classmethod
    def get_config_parameters(cls) -> tuple[str,...]:
        return ("constant_value", )


    def _get_current_value(self, **kwargs: int) -> float:

        assert self.constant_value is not None

        return self.constant_value


class InverseTime(Sequence):

    @classmethod
    def get_config_parameters(cls) -> tuple[str,...]:
        return ("initial_value", )


    def _get_current_value(self, **kwargs: int) -> float:

        if self.decay_parameter not in kwargs:
            raise Exception(f"InverseTime schedule: No value for decay parameter "
                            f"'{self.decay_parameter}' given")

        current_time = kwargs.get(self.decay_parameter)

        assert current_time  is not None  # assert for type checking
        assert self.initial_value  is not None  # assert for type checking

        if current_time == 0:
            value = self.initial_value
        else:
            value = self.initial_value / current_time

        return value


class InverseTimeScaled(Sequence):

    @classmethod
    def get_config_parameters(cls) -> tuple[str,...]:
        return ("initial_value", "n0")


    def _get_current_value(self, **kwargs: int) -> float:

        if self.decay_parameter not in kwargs:
            raise Exception(f"InverseTimeScaled schedule: No value for decay parameter "
                            f"'{self.decay_parameter}' given")

        current_time = kwargs.get(self.decay_parameter)

        assert current_time  is not None  # assert for type checking
        assert self.initial_value  is not None  # assert for type checking
        assert self.n0 is not None  # assert for type checking

        if self.n0 == 0 and current_time == 0:
            value = self.initial_value
        else:
            value = self.initial_value * (self.n0 + 1) / (self.n0 + current_time)

        return value


class InverseTimeTargetAt(Sequence):

    def __init__(self, **kwargs) -> None: # type: ignore[no-untyped-def]

        super().__init__(**kwargs)
        self.n0 = self._get_n0()


    @classmethod
    def get_config_parameters(cls) -> tuple[str,...]:
        return ("initial_value", "target_iterations", "target_value")


    def _get_n0(self) -> float:

        assert self.initial_value is not None  # assert for type checking
        assert self.target_iterations is not None  # assert for type checking
        assert self.target_value is not None  # assert for type checking

        return ((self.initial_value - self.target_iterations*self.target_value) /
               (self.target_value - self.initial_value))


    def _get_current_value(self, **kwargs: int) -> float:

        if self.decay_parameter not in kwargs:
            raise Exception(f"InverseTimeTargetAt schedule: No value for decay parameter "
                            f"'{self.decay_parameter}' given")

        current_time = kwargs.get(self.decay_parameter)

        assert current_time is not None  # assert for type checking
        assert self.target_iterations is not None  # assert for type checking
        assert self.target_value is not None  # assert for type checking
        assert self.initial_value is not None  # assert for type checking
        assert self.n0 is not None  # assert for type checking

        if current_time > self.target_iterations:
            value = self.target_value
        else:
            value = self.initial_value * (self.n0 + 1) / (self.n0 + current_time)

        return value


class Exponential(Sequence):

    @classmethod
    def get_config_parameters(cls) -> tuple[str,...]:
        return ("initial_value", "n0")


    def _get_current_value(self, **kwargs: int) -> float:

        if self.decay_parameter not in kwargs:
            raise Exception(f"Exponential schedule: No value for decay parameter "
                            f"'{self.decay_parameter}' given")

        current_time = kwargs.get(self.decay_parameter)

        assert current_time is not None  # assert for type checking
        assert self.initial_value is not None  # assert for type checking
        assert self.n0 is not None  # assert for type checking

        value = self.initial_value * np.exp(-self.n0 * (current_time - 1)) # pylint: disable=invalid-unary-operand-type

        assert is_scalar_64_tg(value) # -> TypeGuard[np.float64]

        return value.item()


class ExponentialTargetAt(Sequence):

    def __init__(self, **kwargs) -> None: # type: ignore[no-untyped-def]

        super().__init__(**kwargs)
        self.n0 = self._get_n0()


    @classmethod
    def get_config_parameters(cls) -> tuple[str,...]:
        return ("initial_value", "target_iterations", "target_value")


    def _get_n0(self) -> float:

        assert self.initial_value is not None  # assert for type checking
        assert self.target_iterations is not None  # assert for type checking
        assert self.target_value is not None  # assert for type checking

        value = np.log(self.initial_value / self.target_value) / (self.target_iterations - 1)

        assert is_scalar_64_tg(value) # -> TypeGuard[np.float64]

        return value.item()


    def _get_current_value(self, **kwargs: int) -> float:

        if self.decay_parameter not in kwargs:
            raise Exception(f"Exponential schedule: No value for decay parameter "
                            f"'{self.decay_parameter}' given")

        current_time = kwargs.get(self.decay_parameter)

        assert current_time is not None  # assert for type checking
        assert self.target_iterations is not None  # assert for type checking
        assert self.target_value is not None  # assert for type checking
        assert self.initial_value is not None  # assert for type checking
        assert self.n0 is not None  # assert for type checking

        if current_time > self.target_iterations:
            value = self.target_value
        else:
            value = self.initial_value * np.exp(-self.n0 * (current_time - 1)) # pylint: disable=invalid-unary-operand-type

        return value


class SequenceCreator:  # pylint: disable=too-few-public-methods

    schedule_types: dict[str, tuple[Type[Sequence], str]] = {

        "CONSTANT": (Constant, "none"),

        "INV_VISIT_COUNT": (InverseTime, "visit_count"),
        "INV_VISIT_COUNT_SCALED": (InverseTimeScaled, "visit_count"),

        "INV_ROUNDS": (InverseTime, "iteration"),
        "INV_ROUNDS_SCALED": (InverseTimeScaled, "iteration"),
        "INV_ROUNDS_TARGET_AT": (InverseTimeTargetAt, "iteration"),

        "EXPONENTIAL": (Exponential, "iteration"),
        "EXPONENTIAL_TARGET_AT": (ExponentialTargetAt, "iteration"),
    }


    @staticmethod
    def get_schedule(*, schedule_type: str, **kwargs) -> Callable[...,float]: # type: ignore[no-untyped-def] # pylint: disable=line-too-long

        if schedule_type not in SequenceCreator.schedule_types:
            raise Exception(f"Sequence Creator: Unknown schedule type given for get_schedule: "
                            f"{schedule_type}")

        type_def = SequenceCreator.schedule_types.get(schedule_type)

        assert type_def is not None

        sequence_class = type_def[0]
        decay_parameter = type_def[1]

        needed = sequence_class.get_config_parameters()
        given = set(kwargs.keys())

        if not all(x in given for x in needed):
            raise Exception(f"{schedule_type}: Needed parameters for requested schedule type "
                            f"are not available: {needed} vs {given} ")

        if any(kwargs[x] is None for x in needed ):
            raise Exception(f"{schedule_type}: Needed parameters for requested schedule type "
                            f"have missing values: {[( x, kwargs[x]) for x in needed ]}")

        sc_instance = sequence_class(decay_parameter=decay_parameter, **kwargs)

        return sc_instance.get_schedule()


def run_main() -> None:  # pylint: disable=too-many-locals

    schedule_types = list(SequenceCreator.schedule_types.keys())

    target_iterations=90000
    target_value=0.02
    initial_value=0.1
    constant_value=0.1
    n0_rounds = 10000
    n0_exp = 0.0001


    getters = [
        SequenceCreator.get_schedule(schedule_type=schedule_types[0],
                            target_iterations=target_iterations,
                            target_value=target_value, initial_value=initial_value,
                            constant_value=constant_value),
        SequenceCreator.get_schedule(schedule_type=schedule_types[1],
                            target_iterations=target_iterations,
                            target_value=target_value, initial_value=initial_value,
                            constant_value=constant_value),
        SequenceCreator.get_schedule(schedule_type=schedule_types[2],
                            target_iterations=target_iterations,
                            target_value=target_value, initial_value=initial_value,
                            constant_value=constant_value, n0=n0_rounds),
        SequenceCreator.get_schedule(schedule_type=schedule_types[3],
                            target_iterations=target_iterations,
                            target_value=target_value, initial_value=initial_value,
                            constant_value=constant_value),
        SequenceCreator.get_schedule(schedule_type=schedule_types[4],
                            target_iterations=target_iterations,
                            target_value=target_value, initial_value=initial_value,
                            constant_value=constant_value, n0=n0_rounds),
        SequenceCreator.get_schedule(schedule_type=schedule_types[5],
                            target_iterations=target_iterations,
                            target_value=target_value, initial_value=initial_value,
                            constant_value=constant_value),
        SequenceCreator.get_schedule(schedule_type=schedule_types[6],
                            target_iterations=target_iterations,
                            target_value=target_value, initial_value=initial_value,
                            constant_value=constant_value, n0=n0_exp),
        SequenceCreator.get_schedule(schedule_type=schedule_types[7],
                            target_iterations=target_iterations,
                            target_value=target_value, initial_value=initial_value,
                            constant_value=constant_value)
    ]

    i = 0
    print(f"{i:>4}: {schedule_types[0]:>12}{schedule_types[1]:>12}{schedule_types[2]:>12}"
          f"{schedule_types[3]:>12}{schedule_types[4]:>12}"
          f"{schedule_types[5]:>12}{schedule_types[6]:>12}{schedule_types[7]:>12}")

    for i in [1,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]:

        getter_0 = getters[0](iteration=i, visit_count=i)
        getter_1 = getters[1](visit_count=i, iteration=None)
        getter_2 = getters[2](visit_count=i, iteration=None)
        getter_3 = getters[3](iteration=i, visit_count=None)
        getter_4 = getters[4](iteration=i, visit_count=None)
        getter_5 = getters[5](iteration=i, visit_count=None)
        getter_6 = getters[6](iteration=i, visit_count=None)
        getter_7 = getters[7](iteration=i, visit_count=None)

        print(f"{i:>4}: {getter_0:>12,.5f}{getter_1:>12,.5f}{getter_2:>12,.5f}{getter_3:>12,.5f}"
              f"{getter_4:>12,.5f}{getter_5:>12,.5f}{getter_6:>12,.5f}{getter_7:>12,.5f}")


if __name__ == "__main__":
    run_main()
