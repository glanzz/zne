"""
Comprehensive pytest unit tests for ZNEEstimator.

This test suite covers:
- Initialization with valid and invalid parameters
- Default estimator behavior (when no ZNE values are passed)
- Statevector and noisy Aer simulator integration
- Result object format and metadata validation
- Circuit folding functionality
- Extrapolation methods
- Image generation with various flags and directories
- store_scaled_circuits flag behavior
- Downstream compatibility (result consumption by other Qiskit components)
- Noise factor validation (runtime override)
- Precision parameter handling
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers import PrimitiveResult, PubResult, DataBin
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from zne_estimator import ZNEEstimator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_circuit():
    """Create a simple Bell state circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def simple_observable():
    """Create a simple ZZ observable."""
    return SparsePauliOp("ZZ")


@pytest.fixture
def statevector_estimator():
    """Create a StatevectorEstimator instance."""
    return StatevectorEstimator()


@pytest.fixture
def noisy_estimator():
    """Create a noisy Aer estimator."""
    # Wrap in EstimatorV2-compatible interface
    from qiskit_aer.primitives import EstimatorV2
    return EstimatorV2()


@pytest.fixture
def temp_diagram_dir():
    """Create a temporary directory for circuit diagrams."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Test Initialization
# ============================================================================

class TestInitialization:
    """Test ZNEEstimator initialization."""

    def test_init_valid_parameters(self, statevector_estimator):
        """Test initialization with valid parameters."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            extrapolator="linear",
            folding="global",
            default_precision=0.01,
            store_scaled_circuits=True,
            save_circuit_diagrams=False,
        )

        assert zne._base_estimator == statevector_estimator
        assert zne._extrapolator == "linear"
        assert zne._folding == "global"
        assert zne._default_precision == 0.01
        assert zne._store_scaled_circuits is True
        assert zne._save_circuit_diagrams is False

    def test_init_default_parameters(self, statevector_estimator):
        """Test initialization with default parameters."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)

        assert zne._extrapolator == "linear"
        assert zne._folding == "global"
        assert zne._default_precision == 0.0
        assert zne._store_scaled_circuits is True
        assert zne._save_circuit_diagrams is True

    def test_init_with_diagram_saving(self, statevector_estimator, temp_diagram_dir):
        """Test initialization with circuit diagram saving enabled."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            save_circuit_diagrams=True,
            diagram_output_dir=temp_diagram_dir,
        )

        assert zne._save_circuit_diagrams is True
        assert zne._diagram_output_dir == temp_diagram_dir
        assert temp_diagram_dir.exists()

    def test_init_diagram_default_path(self, statevector_estimator):
        """Test initialization with diagram saving but no custom path."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            save_circuit_diagrams=True,
        )

        assert zne._save_circuit_diagrams is True
        assert zne._diagram_output_dir is not None
        # Clean up default directory
        if zne._diagram_output_dir.exists():
            shutil.rmtree(zne._diagram_output_dir, ignore_errors=True)



# ============================================================================
# Test Noise Factor Validation
# ============================================================================

class TestNoiseFactorValidation:
    """Test noise factor validation."""

    def test_validate_valid_odd_integers_global(self, statevector_estimator):
        """Test validation passes for odd integers with global folding."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="global")
        # Should not raise
        zne._validate_noise_factors([1, 3, 5, 7])

    def test_validate_invalid_even_integers_global(self, statevector_estimator):
        """Test validation fails for even integers with global folding."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="global")
        with pytest.raises(ValueError, match="odd integer"):
            zne._validate_noise_factors([1, 2, 3])

    def test_validate_invalid_floats_global(self, statevector_estimator):
        """Test validation fails for non-integer floats with global folding."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="global")
        with pytest.raises(ValueError, match="odd integer"):
            zne._validate_noise_factors([1.0, 3.5, 5.0])

    def test_validate_valid_floats_local(self, statevector_estimator):
        """Test validation passes for float noise factors with local folding."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="local")
        # Should not raise - local folding accepts any factors >= 1
        zne._validate_noise_factors([1.0, 1.5, 2.0, 2.5])

    def test_validate_valid_even_integers_local(self, statevector_estimator):
        """Test validation passes for even integers with local folding."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="local")
        # Should not raise - local folding accepts even integers
        zne._validate_noise_factors([1, 2, 3, 4])

    def test_validate_invalid_negative(self, statevector_estimator):
        """Test validation fails for negative values."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="global")
        with pytest.raises(ValueError, match="must be >= 1"):
            zne._validate_noise_factors([-1, 1, 3])

    def test_validate_invalid_zero(self, statevector_estimator):
        """Test validation fails for zero."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="global")
        with pytest.raises(ValueError, match="must be >= 1"):
            zne._validate_noise_factors([0, 1, 3])

    def test_validate_too_few_factors(self, statevector_estimator):
        """Test validation fails with less than 2 noise factors."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="global")
        with pytest.raises(ValueError, match="at least two"):
            zne._validate_noise_factors([1])


# ============================================================================
# Test Circuit Folding
# ============================================================================

class TestCircuitFolding:
    """Test circuit folding functionality."""

    def test_fold_circuit_factor_1_global(self, statevector_estimator, simple_circuit):
        """Test global folding with noise factor 1 returns original circuit."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="global")
        folded = zne._fold_circuit(simple_circuit, 1.0)

        # Should be a copy with same structure
        assert folded.num_qubits == simple_circuit.num_qubits
        assert len(folded.data) == len(simple_circuit.data)
        assert folded is not simple_circuit  # Should be a copy

    def test_fold_circuit_factor_1_local(self, statevector_estimator, simple_circuit):
        """Test local folding with noise factor 1 returns original circuit."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="local")
        folded = zne._fold_circuit(simple_circuit, 1.0)

        # Should be a copy with same structure
        assert folded.num_qubits == simple_circuit.num_qubits
        assert len(folded.data) == len(simple_circuit.data)
        assert folded is not simple_circuit  # Should be a copy

    def test_fold_circuit_factor_3_global(self, statevector_estimator, simple_circuit):
        """Test global folding with noise factor 3."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="global")
        folded = zne._fold_circuit(simple_circuit, 3.0)

        # Should have metadata
        assert folded.metadata is not None
        assert folded.metadata["zne_folded"] is True
        assert folded.metadata["zne_folding_method"] == "global"
        assert folded.metadata["zne_noise_factor"] == 3.0
        assert folded.metadata["do_not_optimize"] is True

    def test_fold_circuit_factor_2_local(self, statevector_estimator, simple_circuit):
        """Test local folding with noise factor 2."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="local")
        folded = zne._fold_circuit(simple_circuit, 2.0)

        # Should have metadata
        assert folded.metadata is not None
        assert folded.metadata["zne_folded"] is True
        assert folded.metadata["zne_folding_method"] == "local"
        assert folded.metadata["zne_noise_factor"] == 2.0
        assert folded.metadata["do_not_optimize"] is True

        # Local folding with factor 2 should add some gates
        assert len(folded.data) > len(simple_circuit.data)

    def test_fold_circuit_factor_5_global(self, statevector_estimator, simple_circuit):
        """Test global folding with noise factor 5."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="global")
        folded = zne._fold_circuit(simple_circuit, 5.0)

        # Should have more gates than factor 3
        assert folded.metadata["zne_noise_factor"] == 5.0
        assert folded.metadata["zne_folding_method"] == "global"

    def test_folded_circuit_preserves_qubits(self, statevector_estimator, simple_circuit):
        """Test that folding preserves qubit count."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="global")
        folded = zne._fold_circuit(simple_circuit, 3.0)

        # Should preserve number of qubits
        assert folded.num_qubits == simple_circuit.num_qubits

    def test_local_folding_increases_circuit_size(self, statevector_estimator, simple_circuit):
        """Test that local folding increases circuit size proportionally."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="local")

        folded_1_5 = zne._fold_circuit(simple_circuit, 1.5)
        folded_2_0 = zne._fold_circuit(simple_circuit, 2.0)
        folded_3_0 = zne._fold_circuit(simple_circuit, 3.0)

        # Higher noise factors should result in more gates
        assert len(folded_1_5.data) >= len(simple_circuit.data)
        assert len(folded_2_0.data) >= len(folded_1_5.data)
        assert len(folded_3_0.data) >= len(folded_2_0.data)

    def test_invalid_folding_method(self, statevector_estimator, simple_circuit):
        """Test that invalid folding method raises error."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)
        zne._folding = "invalid"

        with pytest.raises(ValueError, match="Unknown folding method"):
            zne._fold_circuit(simple_circuit, 3.0)


# ============================================================================
# Test Extrapolation Methods
# ============================================================================

class TestExtrapolation:
    """Test extrapolation methods."""

    def test_linear_extrapolation(self, statevector_estimator):
        """Test linear extrapolation."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            extrapolator="linear",
        )

        # Linear data: y = 1.0 - 0.1*x, so y(0) = 1.0
        xs = np.array([1.0, 3.0, 5.0])
        ys = np.array([[0.9, 0.7, 0.5]])  # Shape (1, 3)
        stds = np.array([[0.01, 0.01, 0.01]])

        val, std = zne._extrapolate(xs, ys.T, stds.T)

        assert val.shape == (1,)
        assert std.shape == (1,)
        # Should be close to 1.0
        assert abs(val[0] - 1.0) < 0.01

    def test_polynomial_extrapolation(self, statevector_estimator):
        """Test polynomial extrapolation."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            extrapolator="polynomial",
        )

        # Use 4 data points for polynomial (degree 2) to have enough data
        xs = np.array([1.0, 3.0, 5.0, 7.0])
        ys = np.array([[0.9, 0.7, 0.5, 0.3]])
        stds = np.array([[0.01, 0.01, 0.01, 0.01]])

        val, std = zne._extrapolate(xs, ys.T, stds.T)

        assert val.shape == (1,)
        assert std.shape == (1,)
        assert not np.isnan(val[0])
        assert not np.isnan(std[0])

    def test_exponential_extrapolation(self, statevector_estimator):
        """Test exponential extrapolation with positive values."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            extrapolator="exponential",
        )

        # Exponential decay: y = exp(-0.1*x)
        xs = np.array([1.0, 3.0, 5.0])
        ys = np.array([[np.exp(-0.1), np.exp(-0.3), np.exp(-0.5)]])
        stds = np.array([[0.01, 0.01, 0.01]])

        val, std = zne._extrapolate(xs, ys.T, stds.T)

        assert val.shape == (1,)
        assert std.shape == (1,)
        # Should be close to exp(0) = 1.0
        assert abs(val[0] - 1.0) < 0.1

    def test_exponential_fallback_to_linear(self, statevector_estimator):
        """Test exponential extrapolation falls back to linear for non-positive values."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            extrapolator="exponential",
        )

        # Include negative values
        xs = np.array([1.0, 3.0, 5.0])
        ys = np.array([[0.5, -0.1, -0.3]])
        stds = np.array([[0.01, 0.01, 0.01]])

        val, std = zne._extrapolate(xs, ys.T, stds.T)

        # Should not raise, falls back to linear
        assert val.shape == (1,)
        assert std.shape == (1,)

    def test_invalid_extrapolator(self, statevector_estimator):
        """Test that invalid extrapolator raises error during extrapolation."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)
        zne._extrapolator = "invalid"

        xs = np.array([1.0, 3.0, 5.0])
        ys = np.array([0.9, 0.7, 0.5])
        stds = np.array([0.01, 0.01, 0.01])

        with pytest.raises(ValueError, match="Unknown extrapolator"):
            zne._extrapolate_one(xs, ys, stds)


# ============================================================================
# Test Result Object Format
# ============================================================================

class TestResultFormat:
    """Test result object format and metadata."""

    def test_result_structure_basic(self, statevector_estimator, simple_circuit, simple_observable):
        """Test basic result structure."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)
        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3, 5])
        result = job.result()

        # Check result type
        assert isinstance(result, PrimitiveResult)
        assert len(result) == 1

        # Check pub result
        pub_result = result[0]
        assert isinstance(pub_result, PubResult)
        assert hasattr(pub_result, 'data')
        assert hasattr(pub_result, 'metadata')

    def test_result_data_fields(self, statevector_estimator, simple_circuit, simple_observable):
        """Test result data contains evs and stds."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)
        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3, 5])
        result = job.result()

        data = result[0].data
        assert isinstance(data, DataBin)
        assert hasattr(data, 'evs')
        assert hasattr(data, 'stds')
        assert hasattr(data, 'shape')

        # Check data types
        assert isinstance(data.evs, np.ndarray)
        assert isinstance(data.stds, np.ndarray)

    def test_result_metadata_zne_fields(self, statevector_estimator, simple_circuit, simple_observable):
        """Test result metadata contains ZNE-specific fields."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            extrapolator="linear",
        )
        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3, 5])
        result = job.result()

        # Check primitive result metadata
        assert result.metadata is not None
        assert result.metadata['zne'] is True
        assert result.metadata['noise_factors'] == [1, 3, 5]
        assert 'job_id' in result.metadata

        # Check pub result metadata
        metadata = result[0].metadata
        assert 'zne' in metadata
        zne_meta = metadata['zne']

        assert zne_meta['noise_factors'] == [1, 3, 5]
        assert zne_meta['extrapolator'] == 'linear'
        assert 'raw_pub_results' in zne_meta
        assert 'noisy_evs' in zne_meta
        assert 'noisy_stds' in zne_meta
        assert 'target_precision' in zne_meta

    def test_result_scaled_circuits_stored(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that scaled circuits are stored when flag is True."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            store_scaled_circuits=True,
        )
        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3, 5])
        result = job.result()

        zne_meta = result[0].metadata['zne']
        assert 'scaled_circuits' in zne_meta
        assert len(zne_meta['scaled_circuits']) == 3  # 3 noise factors provided
        assert all(isinstance(circ, QuantumCircuit) for circ in zne_meta['scaled_circuits'])

    def test_result_scaled_circuits_not_stored(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that scaled circuits are not stored when flag is False."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            store_scaled_circuits=False,
        )
        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3, 5])
        result = job.result()

        zne_meta = result[0].metadata['zne']
        assert 'scaled_circuits' not in zne_meta


# ============================================================================
# Test Simulator Integration
# ============================================================================

class TestSimulatorIntegration:
    """Test integration with different simulators."""

    def test_statevector_estimator_integration(self, statevector_estimator, simple_circuit, simple_observable):
        """Test with StatevectorEstimator."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)
        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3, 5])
        result = job.result()

        # Should get valid results
        assert len(result) == 1
        assert result[0].data.evs is not None
        # Bell state with ZZ should give ~1.0
        # Handle both 0-d and 1-d arrays
        evs = result[0].data.evs
        ev_value = evs.item() if evs.ndim == 0 else evs[0]
        assert abs(ev_value - 1.0) < 0.1

    def test_noisy_estimator_integration(self, noisy_estimator, simple_circuit, simple_observable):
        """Test with noisy Aer estimator."""
        # Get a backend for transpilation
        from qiskit_aer import AerSimulator
        backend = AerSimulator()

        # Transpile circuit for the backend
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        transpiled_circuit = pm.run(simple_circuit)

        zne = ZNEEstimator(base_estimator=noisy_estimator)
        job = zne.run([(transpiled_circuit, simple_observable)], noise_factors=[1, 3])
        result = job.result()

        # Should get valid results (no custom instructions after removing make_folded_block)
        assert len(result) == 1
        assert result[0].data.evs is not None
        # Result should have valid stds
        stds = result[0].data.stds
        std_value = stds.item() if stds.ndim == 0 else stds[0]
        assert std_value >= 0

    def test_multiple_observables(self, statevector_estimator, simple_circuit):
        """Test with multiple observables."""
        obs1 = SparsePauliOp("ZZ")
        obs2 = SparsePauliOp("XX")
        obs3 = SparsePauliOp("YY")

        zne = ZNEEstimator(base_estimator=statevector_estimator)
        job = zne.run([
            (simple_circuit, obs1),
            (simple_circuit, obs2),
            (simple_circuit, obs3),
        ], noise_factors=[1, 3, 5])
        result = job.result()

        assert len(result) == 3
        assert all(hasattr(r.data, 'evs') for r in result)


# ============================================================================
# Test Image Generation
# ============================================================================

class TestImageGeneration:
    """Test circuit diagram image generation."""

    def test_diagram_generation_disabled(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that diagrams are not saved when flag is False."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            save_circuit_diagrams=False,
        )
        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3, 5])
        result = job.result()

        zne_meta = result[0].metadata['zne']
        assert 'circuit_diagram_paths' not in zne_meta

    def test_diagram_generation_enabled_default_path(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that diagrams are saved to default path when flag is True."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            save_circuit_diagrams=True,
        )

        try:
            job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3, 5])
            result = job.result()

            zne_meta = result[0].metadata['zne']
            assert 'circuit_diagram_paths' in zne_meta

            # Check that paths are present (may be None if matplotlib fails)
            paths = zne_meta['circuit_diagram_paths']
            assert len(paths) == 3  # One per noise factor

            # If paths are not None, verify files exist
            for path_str in paths:
                if path_str is not None:
                    from pathlib import Path as PathLib
                    path = PathLib(path_str)
                    assert path.suffix == '.png'
        finally:
            # Clean up
            if zne._diagram_output_dir and zne._diagram_output_dir.exists():
                shutil.rmtree(zne._diagram_output_dir, ignore_errors=True)

    def test_diagram_generation_custom_directory(self, statevector_estimator, simple_circuit, simple_observable, temp_diagram_dir):
        """Test that diagrams are saved to custom directory."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            save_circuit_diagrams=True,
            diagram_output_dir=temp_diagram_dir,
        )

        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3, 5])
        result = job.result()

        zne_meta = result[0].metadata['zne']
        assert 'circuit_diagram_paths' in zne_meta

        # Check that paths point to custom directory
        paths = zne_meta['circuit_diagram_paths']
        for path_str in paths:
            if path_str is not None:
                from pathlib import Path as PathLib
                path = PathLib(path_str)
                assert temp_diagram_dir in path.parents or path.parent == temp_diagram_dir

    def test_diagram_directory_creation(self, statevector_estimator, temp_diagram_dir):
        """Test that diagram directory is created if it doesn't exist."""
        custom_dir = temp_diagram_dir / "nested" / "diagrams"
        assert not custom_dir.exists()

        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            save_circuit_diagrams=True,
            diagram_output_dir=custom_dir,
        )

        assert custom_dir.exists()


# ============================================================================
# Test Default Estimator Behavior
# ============================================================================

class TestDefaultBehavior:
    """Test that ZNE falls back to base estimator behavior when appropriate."""

    def test_no_noise_factors_uses_base_estimator(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that when no noise_factors are provided, it uses base estimator directly."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)

        # Run without noise_factors - should bypass ZNE
        job = zne.run([(simple_circuit, simple_observable)])
        result = job.result()

        # Result should be valid
        assert len(result) == 1
        assert result[0].data.evs is not None
        # Should NOT have ZNE metadata since we bypassed it
        assert 'zne' not in result.metadata or result.metadata.get('zne') is not True

    def test_noise_factor_minimal(self, statevector_estimator, simple_circuit, simple_observable):
        """Test with minimal valid noise factors (1, 3)."""
        # This tests that minimal ZNE with only 2 factors works
        zne = ZNEEstimator(base_estimator=statevector_estimator)

        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3])
        result = job.result()

        # Result should be valid even with minimal factors
        assert len(result) == 1
        assert result[0].data.evs is not None
        # Verify it used the specified noise factors
        assert result.metadata['noise_factors'] == [1, 3]


# ============================================================================
# Test Runtime Parameter Override
# ============================================================================

class TestRuntimeOverride:
    """Test runtime parameter overrides."""

    def test_noise_factor_runtime_specification(self, statevector_estimator, simple_circuit, simple_observable):
        """Test specifying noise factors at runtime."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)

        # Specify noise factors at runtime
        job = zne.run(
            [(simple_circuit, simple_observable)],
            noise_factors=[1, 3, 5, 7],
        )
        result = job.result()

        # Check that specified factors were used
        assert result.metadata['noise_factors'] == [1, 3, 5, 7]
        assert result[0].metadata['zne']['noise_factors'] == [1, 3, 5, 7]

    def test_precision_override(self, statevector_estimator, simple_circuit, simple_observable):
        """Test overriding precision at runtime."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            default_precision=0.01,
        )

        # Override precision
        job = zne.run(
            [(simple_circuit, simple_observable)],
            noise_factors=[1, 3, 5],
            precision=0.001,
        )
        result = job.result()

        # Check that precision was passed through
        assert result[0].metadata['zne']['target_precision'] == 0.001

    def test_invalid_runtime_noise_factors(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that invalid runtime noise factors raise error."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)

        # Try with invalid factors
        with pytest.raises(ValueError):
            zne.run(
                [(simple_circuit, simple_observable)],
                noise_factors=[1],  # Too few
            )

        with pytest.raises(ValueError):
            zne.run(
                [(simple_circuit, simple_observable)],
                noise_factors=[0.5, 1, 3],  # Less than 1
            )

        with pytest.raises(ValueError):
            zne.run(
                [(simple_circuit, simple_observable)],
                noise_factors=[1, 2, 3],  # Even number
            )


# ============================================================================
# Test Downstream Compatibility
# ============================================================================

class TestDownstreamCompatibility:
    """Test that ZNE results can be consumed by downstream components."""

    def test_result_iteration(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that result can be iterated like standard PrimitiveResult."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)
        job = zne.run([
            (simple_circuit, simple_observable),
            (simple_circuit, SparsePauliOp("XX")),
        ], noise_factors=[1, 3, 5])
        result = job.result()

        # Should be iterable
        pub_results = list(result)
        assert len(pub_results) == 2
        assert all(isinstance(pr, PubResult) for pr in pub_results)

    def test_result_indexing(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that result supports indexing."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)
        job = zne.run([
            (simple_circuit, simple_observable),
            (simple_circuit, SparsePauliOp("XX")),
        ], noise_factors=[1, 3, 5])
        result = job.result()

        # Should support indexing
        assert result[0].data.evs is not None
        assert result[1].data.evs is not None
        assert len(result) == 2

    def test_data_array_operations(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that data arrays support numpy operations."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)
        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3, 5])
        result = job.result()

        evs = result[0].data.evs
        stds = result[0].data.stds

        # Should support numpy operations
        mean_ev = np.mean(evs)
        sum_std = np.sum(stds)

        assert isinstance(mean_ev, (float, np.floating))
        assert isinstance(sum_std, (float, np.floating))
        assert not np.isnan(mean_ev)
        assert not np.isnan(sum_std)

    def test_job_interface(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that job has standard interface."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)
        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 3, 5])

        # Should have result method
        assert hasattr(job, 'result')
        result = job.result()
        assert isinstance(result, PrimitiveResult)


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_pubs(self, statevector_estimator):
        """Test with empty pubs list."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)
        job = zne.run([], noise_factors=[1, 3, 5])
        result = job.result()

        assert len(result) == 0

    def test_single_qubit_circuit(self, statevector_estimator):
        """Test with single-qubit circuit."""
        qc = QuantumCircuit(1)
        qc.h(0)
        obs = SparsePauliOp("Z")

        zne = ZNEEstimator(base_estimator=statevector_estimator)
        job = zne.run([(qc, obs)], noise_factors=[1, 3, 5])
        result = job.result()

        assert len(result) == 1
        # H|0> measured in Z basis should give ~0
        evs = result[0].data.evs
        ev_value = evs.item() if evs.ndim == 0 else evs[0]
        assert abs(ev_value) < 0.1

    def test_large_circuit(self, statevector_estimator):
        """Test with larger circuit."""
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        for i in range(3):
            qc.cx(i, i+1)

        obs = SparsePauliOp("Z" * 4)

        zne = ZNEEstimator(base_estimator=statevector_estimator)
        job = zne.run([(qc, obs)], noise_factors=[1, 3, 5])
        result = job.result()

        assert len(result) == 1
        assert result[0].data.evs is not None

    def test_parameterized_circuit(self, statevector_estimator):
        """Test with parameterized circuit."""
        from qiskit.circuit import Parameter

        theta = Parameter('θ')
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        obs = SparsePauliOp("Z")

        zne = ZNEEstimator(base_estimator=statevector_estimator)

        # Provide parameter values
        job = zne.run([(qc, obs, [np.pi/2])], noise_factors=[1, 3, 5])
        result = job.result()

        assert len(result) == 1
        assert result[0].data.evs is not None

    def test_multiple_parameter_sets(self, statevector_estimator):
        """Test with multiple parameter sets."""
        from qiskit.circuit import Parameter

        theta = Parameter('θ')
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        obs = SparsePauliOp("Z")

        zne = ZNEEstimator(base_estimator=statevector_estimator)

        # Multiple parameter values
        param_values = [[0], [np.pi/4], [np.pi/2]]
        job = zne.run([(qc, obs, param_values)], noise_factors=[1, 3, 5])
        result = job.result()

        assert len(result) == 1
        data_shape = result[0].data.evs.shape
        assert data_shape[0] == 3  # Three parameter sets


# ============================================================================
# Test Precision and Numerical Stability
# ============================================================================

class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_extrapolation_with_small_stds(self, statevector_estimator):
        """Test extrapolation with very small standard deviations."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)

        xs = np.array([1.0, 3.0, 5.0])
        ys = np.array([0.9, 0.7, 0.5])
        stds = np.array([1e-10, 1e-10, 1e-10])  # Very small

        val, std = zne._extrapolate_one(xs, ys, stds)

        # Should not raise or produce NaN
        assert not np.isnan(val)
        assert not np.isnan(std)
        assert np.isfinite(val)
        assert np.isfinite(std)

    def test_extrapolation_with_large_stds(self, statevector_estimator):
        """Test extrapolation with large standard deviations."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)

        xs = np.array([1.0, 3.0, 5.0])
        ys = np.array([0.9, 0.7, 0.5])
        stds = np.array([1.0, 1.0, 1.0])  # Large relative to values

        val, std = zne._extrapolate_one(xs, ys, stds)

        # Should not raise or produce NaN
        assert not np.isnan(val)
        assert not np.isnan(std)
        assert np.isfinite(val)
        assert np.isfinite(std)

    def test_extrapolation_with_zero_stds(self, statevector_estimator):
        """Test extrapolation with zero standard deviations."""
        zne = ZNEEstimator(base_estimator=statevector_estimator)

        xs = np.array([1.0, 3.0, 5.0])
        ys = np.array([0.9, 0.7, 0.5])
        stds = np.array([0.0, 0.0, 0.0])  # Zero

        val, std = zne._extrapolate_one(xs, ys, stds)

        # Should handle gracefully due to clipping
        assert not np.isnan(val)
        assert not np.isnan(std)


# ============================================================================
# Test Local Folding
# ============================================================================

class TestLocalFolding:
    """Test local folding specific functionality."""

    def test_local_folding_with_float_noise_factors(self, statevector_estimator, simple_circuit, simple_observable):
        """Test local folding accepts and works with float noise factors."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            folding="local",
            extrapolator="linear",
        )

        # Run with float noise factors
        job = zne.run(
            [(simple_circuit, simple_observable)],
            noise_factors=[1.0, 1.5, 2.0, 2.5, 3.0]
        )
        result = job.result()

        # Should get valid results
        assert len(result) == 1
        assert result[0].data.evs is not None
        assert result.metadata['noise_factors'] == [1.0, 1.5, 2.0, 2.5, 3.0]

    def test_local_folding_with_even_integers(self, statevector_estimator, simple_circuit, simple_observable):
        """Test local folding accepts even integer noise factors."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            folding="local",
            extrapolator="linear",
        )

        # Run with even integers (not allowed for global folding)
        job = zne.run(
            [(simple_circuit, simple_observable)],
            noise_factors=[1, 2, 3, 4]
        )
        result = job.result()

        # Should succeed
        assert len(result) == 1
        assert result[0].data.evs is not None

    def test_local_folding_circuit_structure(self, statevector_estimator, simple_circuit):
        """Test local folding produces correct circuit structure."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="local")

        # Test with a known circuit
        folded = zne._fold_circuit(simple_circuit, 2.0)

        # Check that some gates were folded (circuit should be larger)
        assert len(folded.data) > len(simple_circuit.data)

        # Check metadata
        assert folded.metadata["zne_folding_method"] == "local"
        assert folded.metadata["zne_noise_factor"] == 2.0

    def test_local_folding_empty_circuit(self, statevector_estimator):
        """Test local folding handles empty circuits gracefully."""
        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="local")

        empty_circuit = QuantumCircuit(2)
        folded = zne._fold_circuit(empty_circuit, 2.0)

        # Should handle empty circuit without error
        assert folded.num_qubits == 2
        assert len(folded.data) == 0

    def test_local_folding_preserves_register_names(self, statevector_estimator):
        """Test that local folding preserves quantum and classical register names."""
        from qiskit import QuantumRegister, ClassicalRegister

        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])

        zne = ZNEEstimator(base_estimator=statevector_estimator, folding="local")
        folded = zne._fold_circuit(qc, 2.0)

        # Check register names preserved
        assert len(folded.qregs) == 1
        assert len(folded.cregs) == 1
        assert folded.qregs[0].name == 'q'
        assert folded.cregs[0].name == 'c'

    def test_local_vs_global_folding_results(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that local and global folding both produce valid results."""
        zne_global = ZNEEstimator(
            base_estimator=statevector_estimator,
            folding="global",
            extrapolator="linear",
        )

        zne_local = ZNEEstimator(
            base_estimator=statevector_estimator,
            folding="local",
            extrapolator="linear",
        )

        # Use odd integers for both
        noise_factors = [1, 3, 5]

        job_global = zne_global.run([(simple_circuit, simple_observable)], noise_factors=noise_factors)
        result_global = job_global.result()

        job_local = zne_local.run([(simple_circuit, simple_observable)], noise_factors=noise_factors)
        result_local = job_local.result()

        # Both should produce valid results
        assert result_global[0].data.evs is not None
        assert result_local[0].data.evs is not None

        # Results should be reasonably close (but not identical due to different folding strategies)
        ev_global = result_global[0].data.evs.item() if result_global[0].data.evs.ndim == 0 else result_global[0].data.evs[0]
        ev_local = result_local[0].data.evs.item() if result_local[0].data.evs.ndim == 0 else result_local[0].data.evs[0]

        # Both should be close to ideal value (Bell state ZZ = 1.0)
        assert abs(ev_global - 1.0) < 0.2
        assert abs(ev_local - 1.0) < 0.2

    def test_local_folding_metadata_in_results(self, statevector_estimator, simple_circuit, simple_observable):
        """Test that local folding metadata is correctly stored in results."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            folding="local",
            store_scaled_circuits=True,
        )

        job = zne.run([(simple_circuit, simple_observable)], noise_factors=[1, 2, 3])
        result = job.result()

        # Check metadata
        zne_meta = result[0].metadata['zne']
        assert 'scaled_circuits' in zne_meta

        # Check each scaled circuit has correct metadata
        for circuit in zne_meta['scaled_circuits']:
            if circuit.metadata and 'zne_folded' in circuit.metadata:
                assert circuit.metadata['zne_folding_method'] == 'local'

    def test_local_folding_fine_grained_noise_control(self, statevector_estimator, simple_circuit, simple_observable):
        """Test local folding with fine-grained noise factor control."""
        zne = ZNEEstimator(
            base_estimator=statevector_estimator,
            folding="local",
            extrapolator="polynomial",
        )

        # Use many fine-grained noise factors
        noise_factors = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        job = zne.run([(simple_circuit, simple_observable)], noise_factors=noise_factors)
        result = job.result()

        # Should produce valid results with fine-grained control
        assert len(result) == 1
        assert result[0].data.evs is not None
        assert result.metadata['noise_factors'] == noise_factors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
