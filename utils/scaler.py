from typing import Callable, Optional, NamedTuple


class MinMaxScale(NamedTuple):
    scale_min: float
    scale_max: float
    coeff_a: Optional[float]
    coeff_b: Optional[float]

class Scaler:
    """Scales values linearly from interval [min,max} to [0,1]"""

    def __init__(self) -> None:
        self._scales: dict[str, MinMaxScale] = {}

    def register_scale(self, key: str, scale_min: float, scale_max: float) -> None:

        if key in self._scales:
            if scale_min == self._scales[key].scale_min or scale_max == self._scales[key].scale_max:
                return
            else:
                raise SystemExit(f"Scaler: trying to register a scale {key} with conflicting values: "
                                 f"min {scale_min} vs {self._scales[key].scale_min} and "
                                 f"max {scale_max} vs {self._scales[key].scale_max}")

        if scale_min == scale_max:
            coeff_a = None
            coeff_b = None
        else:
            coeff_a = 1 / (scale_max - scale_min)
            coeff_b = scale_min / (scale_max - scale_min)

        self._scales[key] = MinMaxScale(scale_min, scale_max, coeff_a, coeff_b)


    def get_scale(self, key: str) -> MinMaxScale:
        return self._scales[key]


    def scale_value(self, value: float, key: str) -> float:

        coeff_a = self._scales[key].coeff_a
        coeff_b = self._scales[key].coeff_b

        if coeff_a is None or coeff_b is None:
            return 1

        return coeff_a * value - coeff_b


    def get_scaler(self) -> Callable[[float, str], float]:
        return self.scale_value


scaler = Scaler()
