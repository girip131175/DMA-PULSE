"""Two-stage AOA/RFO-style hyperparameter search utilities.

The paper specifies the optimization roles and search variables, but does not
publish enough pseudocode to claim this module is an exact reproduction of the
authors' optimizer implementation. The routines below are therefore an
explicit engineering implementation of the described global-then-local search
pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

Objective = Callable[[Mapping[str, float]], float]


@dataclass(frozen=True)
class SearchParameter:
    name: str
    low: float
    high: float
    integer: bool = False


@dataclass
class SearchResult:
    params: dict[str, float]
    score: float


def _decode(vector: np.ndarray, space: Sequence[SearchParameter]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value, spec in zip(vector, space):
        clipped = float(np.clip(value, spec.low, spec.high))
        result[spec.name] = int(round(clipped)) if spec.integer else clipped
    return result


class AOAOptimizer:
    """Global exploration stage for the DMA-PULSE search pipeline."""

    def __init__(self, space: Sequence[SearchParameter], population: int = 8, seed: int = 42):
        self.space = list(space)
        self.population = population
        self.rng = np.random.default_rng(seed)

    def optimize(self, objective: Objective, iterations: int = 10) -> SearchResult:
        lows = np.array([p.low for p in self.space], dtype=float)
        highs = np.array([p.high for p in self.space], dtype=float)
        pop = self.rng.uniform(lows, highs, size=(self.population, len(self.space)))
        best_vector = pop[0].copy()
        best_score = float("inf")

        for iteration in range(iterations):
            progress = (iteration + 1) / max(iterations, 1)
            for vector in pop:
                score = float(objective(_decode(vector, self.space)))
                if score < best_score:
                    best_score = score
                    best_vector = vector.copy()

            # Exploration contracts toward the best candidate while a small
            # stochastic term maintains global coverage.
            scale = 1.0 - progress
            noise = self.rng.normal(0.0, 1.0, size=pop.shape)
            attraction = self.rng.uniform(0.0, 1.0, size=pop.shape) * (best_vector - pop)
            pop = np.clip(
                pop + scale * attraction + 0.05 * scale * noise * (highs - lows),
                lows,
                highs,
            )

        return SearchResult(_decode(best_vector, self.space), best_score)


class RFOOptimizer:
    """Local refinement stage seeded by the AOA search result."""

    def __init__(self, space: Sequence[SearchParameter], population: int = 6, seed: int = 123):
        self.space = list(space)
        self.population = population
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        objective: Objective,
        seed_result: SearchResult,
        iterations: int = 8,
        radius: float = 0.15,
    ) -> SearchResult:
        lows = np.array([p.low for p in self.space], dtype=float)
        highs = np.array([p.high for p in self.space], dtype=float)
        span = highs - lows
        seed_params = seed_result.params
        center = np.array(
            [seed_params.get(p.name, (p.low + p.high) / 2.0) for p in self.space],
            dtype=float,
        )
        best_vector = center.copy()
        best_score = seed_result.score

        for iteration in range(iterations):
            local_radius = radius * (1.0 - iteration / max(iterations, 1))
            candidates = center + self.rng.normal(
                0.0, local_radius, size=(self.population, len(self.space))
            ) * span
            candidates = np.clip(candidates, lows, highs)

            for vector in candidates:
                score = float(objective(_decode(vector, self.space)))
                if score < best_score:
                    best_score = score
                    best_vector = vector.copy()
                    center = vector.copy()

        return SearchResult(_decode(best_vector, self.space), best_score)


class DMAPulseOptimizer:
    """Convenience wrapper implementing the published AOA→RFO sequence."""

    def __init__(
        self,
        global_space: Sequence[SearchParameter],
        local_space: Sequence[SearchParameter],
        seed: int = 42,
    ):
        self.global_optimizer = AOAOptimizer(global_space, seed=seed)
        self.local_optimizer = RFOOptimizer(local_space, seed=seed + 1)

    def optimize(self, objective: Objective) -> tuple[SearchResult, SearchResult]:
        global_result = self.global_optimizer.optimize(objective)
        local_result = self.local_optimizer.optimize(objective, global_result)
        return global_result, local_result
