import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def reset(self, seed: int | None = None) -> None:
        """Reset the policy to its initial state."""
        pass
