from __future__ import annotations

import numpy as np


class BudgetAwareDecoder:
    """Threshold + deterministic preference-per-token repair from the manuscript."""

    def __init__(self, costs: np.ndarray, budget: int, threshold: float = 0.5):
        self.costs = np.asarray(costs, dtype=np.int64)
        self.budget = int(budget)
        self.threshold = float(threshold)
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if np.any(self.costs <= 0):
            raise ValueError("all context units must have positive token costs")

    def decode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or len(x) != len(self.costs):
            raise ValueError("preference vector has wrong shape")
        mask = x >= self.threshold
        ratio = x / self.costs
        used = int(self.costs[mask].sum())

        if used > self.budget:
            selected = np.flatnonzero(mask)
            # Drop low preference-per-token first; stable tie break by index.
            order = selected[np.argsort(ratio[selected], kind="mergesort")]
            for j in order.tolist():
                if used <= self.budget:
                    break
                mask[j] = False
                used -= int(self.costs[j])

        # Fill remaining capacity from high ratio unselected candidates.
        unselected = np.flatnonzero(~mask)
        order = unselected[np.argsort(-ratio[unselected], kind="mergesort")]
        for j in order.tolist():
            c = int(self.costs[j])
            if used + c <= self.budget:
                mask[j] = True
                used += c
        return mask

    def decode_batch(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("batch preferences must have shape [N, m]")
        return np.stack([self.decode(row) for row in x], axis=0)

    def feasible(self, mask: np.ndarray) -> bool:
        return int(self.costs[np.asarray(mask, dtype=bool)].sum()) <= self.budget


class BallConstrainedDecoder:
    """Threshold inside the k-anchor radius-d feasible set + ΔF-guided repair.

    Admission is a membership test rather than a reachability test, so dropping a
    unit never invalidates another one. Repair is therefore a plain exchange:
    shed the worst fitness-per-token unit while over budget, then absorb the best
    one that still fits.
    """

    def __init__(
        self,
        member: np.ndarray,
        costs: np.ndarray,
        budget: int,
        fitness,
        threshold: float = 0.5,
        repair: bool = True,
    ):
        self.member = np.asarray(member, dtype=bool)
        self.costs = np.asarray(costs, dtype=np.int64)
        self.budget = int(budget)
        self.fitness = fitness
        self.threshold = float(threshold)
        self.repair = bool(repair)
        self.m = len(self.costs)
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if np.any(self.costs <= 0):
            raise ValueError("all context units must have positive token costs")
        if len(self.member) != self.m:
            raise ValueError("membership mask and costs must have equal length")

    def decode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or len(x) != self.m:
            raise ValueError("preference vector has wrong shape")
        selected = (x >= self.threshold) & self.member
        if not self.repair:
            return selected
        used = int(self.costs[selected].sum())
        if used > self.budget:
            selected = self._shed(selected)
        # Feasibility is not optimality: under a token price the objective may
        # prefer a smaller context, so drop whatever no longer pays for itself.
        selected = self._trim(selected)
        selected = self._absorb(selected)
        return selected

    def _move_scores(self, selected: np.ndarray, moves: np.ndarray, value: bool) -> np.ndarray:
        """Exact fitness of every one-unit variation of `selected`, in one batch.

        Each candidate move is scored against the same base subset, so the moves
        are independent of one another and evaluating them together is identical
        to evaluating them one at a time. Only the surrounding greedy loop, which
        commits a move before proposing the next one, has to stay sequential.
        """
        trials = np.repeat(selected[None, :], len(moves), axis=0)
        trials[np.arange(len(moves)), moves] = value
        return np.asarray(self.fitness.evaluate_batch(trials), dtype=np.float64)

    def _trim(self, selected: np.ndarray) -> np.ndarray:
        selected = selected.copy()
        while int(selected.sum()) > 1:
            heads = np.flatnonzero(selected)
            current = float(self.fitness.evaluate(selected))
            gains = self._move_scores(selected, heads, False) - current
            eligible = np.flatnonzero(gains > 0)
            if len(eligible) == 0:
                break
            # argmax takes the first maximum and `heads` is ascending, so ties
            # resolve to the lowest index, as the sequential scan did.
            selected[heads[eligible[int(np.argmax(gains[eligible]))]]] = False
        return selected

    def _shed(self, selected: np.ndarray) -> np.ndarray:
        selected = selected.copy()
        used = int(self.costs[selected].sum())
        while used > self.budget:
            heads = np.flatnonzero(selected)
            if len(heads) == 0:
                break
            current = float(self.fitness.evaluate(selected))
            losses = current - self._move_scores(selected, heads, False)
            j = int(heads[int(np.argmin(losses / self.costs[heads]))])
            selected[j] = False
            used -= int(self.costs[j])
        return selected

    def _absorb(self, selected: np.ndarray) -> np.ndarray:
        selected = selected.copy()
        used = int(self.costs[selected].sum())
        while used < self.budget:
            cand = np.flatnonzero(self.member & ~selected)
            costs = self.costs[cand]
            affordable = used + costs <= self.budget
            cand, costs = cand[affordable], costs[affordable]
            if len(cand) == 0:
                break
            current = float(self.fitness.evaluate(selected))
            deltas = self._move_scores(selected, cand, True) - current
            eligible = np.flatnonzero(deltas > 0)
            if len(eligible) == 0:
                break
            pick = eligible[int(np.argmax(deltas[eligible] / costs[eligible]))]
            selected[cand[pick]] = True
            used += int(costs[pick])
        return selected

    def decode_batch(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("batch preferences must have shape [N, m]")
        return np.stack([self.decode(row) for row in x], axis=0)

    def feasible(self, mask: np.ndarray) -> bool:
        mask = np.asarray(mask, dtype=bool)
        if np.any(mask & ~self.member):
            return False
        return int(self.costs[mask].sum()) <= self.budget
