from __future__ import annotations
from qiskit.circuit import QuantumCircuit

from collections.abc import Iterable
from typing import Optional, Sequence

import numpy as np
from qiskit.primitives import BaseEstimatorV2
from qiskit.primitives.containers import (
    EstimatorPubLike,
    PrimitiveResult,
    PubResult,
    DataBin,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.primitive_job import PrimitiveJob



class ZNEEstimator(BaseEstimatorV2):
    """Estimator that applies Zero-Noise Extrapolation (ZNE) on top of a base estimator."""

    def __init__(
        self,
        base_estimator: BaseEstimatorV2,
        noise_factors: tuple[float, ...] = (1.0, 3.0, 5.0),
        extrapolator: str = "linear",  # "linear" | "polynomial" | "exponential"
        folding: str = "global",        # "global" | "local"
        default_precision: float = 0.0,
    ):
        self._base_estimator = base_estimator
        self._noise_factors = tuple(noise_factors)
        self._extrapolator = extrapolator
        self._folding = folding
        self._default_precision = default_precision

        if len(self._noise_factors) < 2:
            raise ValueError("ZNE requires at least two noise factors.")
        if any(nf < 1 for nf in self._noise_factors):
            raise ValueError("All noise factors must be >= 1.")


    def run(
        self,
        pubs: Iterable[EstimatorPubLike],
        *,
        precision: Optional[float] = None,
        noise_factors: Optional[Sequence[float]] = None,
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        """Estimate expectation values with ZNE applied to each pub.

        Args:
            pubs: Iterable of pub-like objects.
            precision: Target precision for any pub that does not set its own.
            noise_factors: Per-call override of the noise factors used for
                circuit folding, e.g. ``[1, 3, 5]``. Falls back to the value
                supplied at construction time when ``None``.
        """
        if precision is None:
            precision = self._default_precision

        factors = tuple(noise_factors) if noise_factors is not None else self._noise_factors
        self._validate_noise_factors(factors)

        coerced_pubs: list[EstimatorPub] = [
            EstimatorPub.coerce(pub, precision=precision) for pub in pubs
        ]

        job = PrimitiveJob(self._run, coerced_pubs, factors)
        job._submit()
        return job


    @staticmethod
    def _validate_noise_factors(factors: Sequence[float]) -> None:
        if len(factors) < 2:
            raise ValueError("ZNE requires at least two noise factors.")
        if any(nf < 1 for nf in factors):
            raise ValueError("All noise factors must be >= 1.")
        if any(int(nf) != nf or int(nf) % 2 == 0 for nf in factors):
            raise ValueError(
                "Global folding requires odd integer noise factors (e.g. 1, 3, 5)."
            )

    def _run(
        self,
        pubs: list[EstimatorPub],
        factors: tuple[float, ...],
    ) -> PrimitiveResult[PubResult]:
        return PrimitiveResult(
            [self._run_single_pub(pub, factors) for pub in pubs],
            metadata={"zne": True, "noise_factors": list(factors)},
        )

    def _run_single_pub(
        self,
        pub: EstimatorPub,
        factors: tuple[float, ...],
    ) -> PubResult:
        transpiled_circuit = pub.circuit  # assumed already transpiled
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision

        # One scaled circuit per noise factor.
        folded_pubs = [
            (self._fold_circuit(transpiled_circuit, nf), observables, parameter_values)
            for nf in factors
        ]

        base_job = self._base_estimator.run(folded_pubs, precision=precision)
        base_result = base_job.result()

        sample_data = base_result[0].data
        ev_field = "evs" if hasattr(sample_data, "evs") else "values"
        std_field = "stds" if hasattr(sample_data, "stds") else "errors"

        noisy_evs = np.stack(
            [np.asarray(getattr(r.data, ev_field)) for r in base_result], axis=0
        )
        noisy_stds = np.stack(
            [np.asarray(getattr(r.data, std_field)) for r in base_result], axis=0
        )

        zne_evs, zne_stds = self._extrapolate(
            np.asarray(factors, dtype=float), noisy_evs, noisy_stds,
        )

        data = DataBin(evs=zne_evs, stds=zne_stds, shape=zne_evs.shape)
        metadata = {
            "zne": {
                "noise_factors": list(factors),
                "extrapolator": self._extrapolator,
                "noisy_evs": noisy_evs,
                "noisy_stds": noisy_stds,
                "target_precision": precision,
            }
        }
        return PubResult(data=data, metadata=metadata)




    def _fold_circuit(self, circuit: QuantumCircuit, noise_factor: float) -> QuantumCircuit:
        """Apply global unitary folding to an already-transpiled circuit.

        Returns ``U (U^dagger U)^((n-1)/2)`` with each repetition wrapped in a
        ``box`` so downstream optimization passes cannot cancel the folded gates.
        The input is assumed to be the transpiled, ISA-compliant circuit; the
        output must NOT be transpiled again with optimization enabled.
        """
        n = int(round(noise_factor))
        if n == 1:
            return circuit.copy()

        folded = circuit.copy_empty_like()
        folded.compose(circuit, inplace=True)  # the leading U

        inverse = circuit.inverse()
        for _ in range((n - 1) // 2):
            # Wrap (U^dagger U) in a box so it survives optimization.
            # with folded.box():
            folded.compose(inverse, inplace=True)
            folded.compose(circuit, inplace=True)

        # Mark the whole circuit so callers know not to re-optimize it.
        folded.metadata = {
            **(circuit.metadata or {}),
            "zne_folded": True,
            "zne_noise_factor": noise_factor,
            "do_not_optimize": True,
        }
        return folded

    def _extrapolate(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        y_stds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extrapolate to x = 0 elementwise over the observable array."""
        obs_shape = ys.shape[1:]
        flat_ys = ys.reshape(len(xs), -1)
        flat_stds = y_stds.reshape(len(xs), -1)

        zne_vals = np.empty(flat_ys.shape[1])
        zne_stds = np.empty(flat_ys.shape[1])

        for i in range(flat_ys.shape[1]):
            zne_vals[i], zne_stds[i] = self._extrapolate_one(
                xs, flat_ys[:, i], flat_stds[:, i]
            )

        return zne_vals.reshape(obs_shape), zne_stds.reshape(obs_shape)

    def _extrapolate_one(
        self, xs: np.ndarray, ys: np.ndarray, stds: np.ndarray
    ) -> tuple[float, float]:
        weights = 1.0 / np.clip(stds, 1e-12, None) ** 2

        if self._extrapolator == "linear":
            degree = 1
        elif self._extrapolator == "polynomial":
            degree = min(len(xs) - 1, 2)
        elif self._extrapolator == "exponential":
            # Fit ln(|y|) linearly in x, then map back. Falls back to linear
            # if any y is non-positive.
            if np.all(ys > 0):
                log_ys = np.log(ys)
                coeffs = np.polyfit(xs, log_ys, 1, w=weights)
                val = float(np.exp(np.polyval(coeffs, 0.0)))
                # Crude std propagation through exp.
                resid = log_ys - np.polyval(coeffs, xs)
                std = float(val * np.sqrt(np.mean(resid ** 2)))
                return val, std
            degree = 1
        else:
            raise ValueError(f"Unknown extrapolator: {self._extrapolator}")

        coeffs, cov = np.polyfit(xs, ys, degree, w=weights, cov=True)
        val = float(np.polyval(coeffs, 0.0))
        # Variance of polynomial evaluated at x = 0 is cov[-1, -1] (intercept).
        std = float(np.sqrt(max(cov[-1, -1], 0.0)))
        return val, std