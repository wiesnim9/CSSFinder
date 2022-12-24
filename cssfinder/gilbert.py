from __future__ import annotations

import logging
# from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional

import numpy as np

from cssfinder import ops
from cssfinder.log import get_logger
from cssfinder.modes import Mode, ModeABC
from cssfinder.types import MtxC128T, MtxT


class Gilbert:
    """Gilbert algorithm implementation."""

    def __init__(
        self,
        mode: Mode,
        initial_state: MtxT,
        size: Optional[int],
        sub_sys_size: Optional[int],
    ) -> None:
        self.mode = ModeABC.use(mode)()
        self.initial_state = initial_state
        self.logger = get_logger()

        self.logger.debug("Created Gilbert algorithm using mode: {!r}", self.mode)
        self.logger.debug(
            "Using initial matrix: shape {!r} dtype {}",
            self.initial_state.shape,
            self.initial_state.dtype,
        )

        if sub_sys_size is None:
            if size is None:
                (
                    self.size,
                    self.sub_sys_size,
                ) = self.mode.detect_dims_none_given(self.total_system_size)
            else:
                (
                    self.size,
                    self.sub_sys_size,
                ) = self.mode.detect_dims_size_given(size, self.total_system_size)

    @property
    def total_system_size(self) -> int:
        """Total size of system determined from initial state first axis size."""
        return len(self.initial_state)

    def run(self, visibility: float, steps: int, correlations: int) -> None:

        total_sys_size = self.total_system_size
        identity_mtx = np.identity(total_sys_size)
        state = self.initial_state
        inverse_visibility = np.subtract(1, visibility)

        rho = np.add(
            np.multiply(visibility, state),
            np.divide(
                np.multiply(inverse_visibility, identity_mtx),
                total_sys_size,
            ),
        )

        start_time = perf_counter()
        _gilbert(
            rho.astype(np.complex128),
            self.mode,
            steps,
            correlations,
            self.size,
            self.sub_sys_size,
            0.0000001,
        )
        end_time = perf_counter()
        logging.critical(f"Optimization took {end_time - start_time:.0f}s")


@dataclass
class Correlation:

    iter_left: int
    found_at_iter: int
    value: np.float64


def _gilbert(
    rho: MtxC128T,
    mode: ModeABC,
    steps: int,
    correlations: int,
    size: int,
    sub_sys_size: int,
    precision: float,
    log_every_epochs: int = 5000,
) -> None:

    # executor = ThreadPoolExecutor(max_workers=4)

    logger = get_logger()
    logger.debug("==================")
    logger.debug(" _gilbert params:")
    logger.debug("==================")
    logger.debug("  mode            = {}", mode)
    logger.debug("  steps           = {}", steps)
    logger.debug("  correlations    = {}", correlations)
    logger.debug("  size            = {}", size)
    logger.debug("  sub_sys_size    = {}", sub_sys_size)
    logger.debug("  precision       = {}", precision)
    logger.debug("==================")

    _debug_msg_short_rho(True, 0, rho)

    rho1 = np.zeros_like(rho, dtype=np.complex128)
    np.fill_diagonal(rho1, rho.diagonal())

    _debug_msg_short_rho(True, 1, rho1)

    rho3 = rho - rho1
    _debug_msg_short_rho(True, 3, rho3)

    # product_0_1, product_1_1, product_1_3 = executor.map(
    #     ops.product, [rho, rho1, rho1], [rho1, rho1, rho3]
    # )

    product_0_1 = ops.product(rho, rho1)
    logger.debug("Product RHO0 RHO1 type: {} value: {}", type(product_0_1), product_0_1)

    product_1_1 = ops.product(rho1, rho1)
    logger.debug("Product RHO0 RHO1 type: {} value: {}", type(product_1_1), product_1_1)

    product_1_3 = ops.product(rho1, rho3)
    logger.debug("Product RHO1 RHO3 type: {} value: {}", type(product_1_3), product_1_3)

    optimization_epochs = 20 * size * size * sub_sys_size

    correlations_list: list[Correlation] = []
    idx = 0

    limiter_product_1_3 = product_1_3
    logger.info("Starting optimization...")

    for idx in range(steps):
        is_log_iter = bool(idx % log_every_epochs == 0)

        if is_log_iter:
            logger.debug("Optimization epoch: {}, product_1_3: {}", idx, product_1_3)

        if len(correlations_list) >= correlations:
            return

        if correlations_list and correlations_list[-1].value <= precision:
            return

        rho2 = mode.random(size, sub_sys_size)
        _debug_msg_short_rho(is_log_iter, 2, rho2)

        product_2_3 = ops.product(rho2, rho3)
        _debug_msg_product(is_log_iter, 2, 3, product_2_3)
        _debug_msg_product(is_log_iter, 1, 3, limiter_product_1_3)

        if product_2_3 > limiter_product_1_3:
            if is_log_iter:
                logger.debug("Optimization epoch {}", product_2_3)

            rho2 = mode.optimize(rho2, rho3, size, sub_sys_size, optimization_epochs)

            # product_0_2, product_1_2, product_2_2 = executor.map(
            #     ops.product, [rho, rho1, rho2], [rho2, rho2, rho2]
            # )

            product_0_2 = ops.product(rho, rho2)
            _debug_msg_product(is_log_iter, 0, 2, product_0_2)
            double_product_0_2 = 2 * product_0_2

            product_1_2 = ops.product(rho1, rho2)
            _debug_msg_product(is_log_iter, 1, 2, product_1_2)
            double_product_1_2 = 2 * product_1_2

            product_2_2 = ops.product(rho2, rho2)
            _debug_msg_product(is_log_iter, 2, 2, product_2_2)
            double_product_2_2 = 2 * product_2_2

            bb2 = (
                -product_0_1
                + double_product_0_2
                + double_product_1_2
                - double_product_2_2
            )
            bb3 = product_1_1 - double_product_1_2 + product_2_2
            cc1 = -bb2 / (2 * bb3)

            if 0 < cc1 <= 1:
                logger.debug(f"Altered statue with cc1 {cc1}")
                rho1 = cc1 * rho1 + (1 - cc1) * rho2

                rho3 = rho - rho1

                product_1_1 = ops.product(rho1, rho1)

                product_1_3 = product_0_1 - product_1_1
                limiter_product_1_3 = product_1_3

                double_product_0_1 = 2 * product_0_1
                product_0_1 = double_product_0_1

    logger.info("Finished optimization...")
    # executor.__exit__(None, None, None)


def _debug_msg_product(is_log_iter: bool, x: int, y: int, prod: float) -> None:
    if is_log_iter:
        get_logger().debug("Product RHO{} RHO{}: {}, type: {}", x, y, prod, type(prod))


def _debug_msg_short_rho(is_log_iter: bool, x: int, rho: Any) -> None:
    if is_log_iter:
        get_logger().debug(
            "\n  RHO{}  type: {}  shape: {}  dtype: {}",
            x,
            type(rho),
            rho.shape,
            rho.dtype,
        )
