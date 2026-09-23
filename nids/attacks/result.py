"""Standardized attack result container."""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AttackResult:
    """Result of running an adversarial attack on one or more samples."""

    x_adv: np.ndarray
    x_clean: np.ndarray
    success: np.ndarray | bool
    n_queries: int = 0
    runtime_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def evasion_rate(self) -> float:
        s = np.asarray(self.success)
        return float(s.mean()) if s.size else 0.0

    def __getitem__(self, i):
        return AttackResult(
            x_adv=self.x_adv[i:i+1],
            x_clean=self.x_clean[i:i+1],
            success=self.success[i] if isinstance(self.success, np.ndarray) else self.success,
            n_queries=self.n_queries,
            runtime_ms=self.runtime_ms,
            metadata=self.metadata,
        )
