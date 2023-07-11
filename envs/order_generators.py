from abc import ABC, abstractmethod
from typing import TypeVar, List

import numpy as np

OrderType = TypeVar("OrderType")


class OrderGenerator(ABC):
    """Abstract base class for generating an order."""
    @abstractmethod
    def should_make_order(self, num_current_orders: int,  elapsed_steps=1) -> bool:
        """Check whether a new order should be made."""
        raise NotImplementedError

    @abstractmethod
    def make_order(self, size) -> OrderType:
        """Make an order."""
        raise NotImplementedError


class BinomialOrderGenerator(OrderGenerator):
    """Order generator that that has a binomial probability for each item."""

    def should_make_order(self, num_current_orders: int,  elapsed_steps=1) -> bool:
        return bool(np.random.binomial(1, 0.1)) or num_current_orders == 0

    def make_order(self, size) -> OrderType:
        order = np.zeros(size, dtype=int)
        while sum(order) == 0:
            order = (np.random.binomial(1, 0.5, size=size)).astype(int)

        return order


class GaussianOrderGenerator(OrderGenerator):

    def __init__(self, p: List[float], max_order_size: int, generator=None) -> None:
        super().__init__()
        if generator is None:
            generator = np.random.default_rng()
        self._random = generator
        p = np.array(p)
        self._p = p / p.sum()
        self._max = max_order_size

    def set_seed(self, seed: int) -> None:
        self._random = np.random.default_rng(seed)

    def should_make_order(self, num_current_orders: int, elapsed_steps=1) -> bool:
        if elapsed_steps < 1:
            elapsed_steps = 1
        return bool(self._random.binomial(elapsed_steps, 0.1)) or num_current_orders == 0

    def make_order(self, size) -> OrderType:
        order = np.zeros(size, dtype=int)
        num_items = np.floor(np.random.triangular(1, self._max / 2, self._max)).astype(int)
        items = self._random.choice(size, size=num_items, p=self._p, replace=False)
        order[items] = 1
        return order


class StaticOrderGenerator(OrderGenerator):
    """
    Repeated provides one pre-defined order. Will always keep a max of one order in circulation.
    Good for debugging or demonstration purposes.
    """

    def __init__(self, order: np.ndarray) -> None:
        super().__init__()
        self._order = order

    def should_make_order(self, num_current_orders: int,  elapsed_steps=1) -> bool:
        return num_current_orders == 0

    def make_order(self, size) -> OrderType:
        assert len(self._order) == size
        return self._order


class MockOrderGenerator(OrderGenerator):
    """A fake order generator for testing!"""

    def __init__(self, orders: List) -> None:
        super().__init__()
        self._orders = orders

    def should_make_order(self, num_current_orders: int, elapsed_steps=1) -> bool:
        if len(self._orders) == 0:
            return False

        pending_order = self._orders[0]
        if pending_order is None:
            self._orders.pop(0)
            return False
        else:
            return True

    def make_order(self, size) -> OrderType:
        assert len(self._orders) != 0
        order = self._orders.pop(0)
        assert order is not None
        assert len(order) == size
        return order
