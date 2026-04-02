"""
Unit tests for observation utilities (padding/truncation)
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.utils_observation import pad_or_trunc


def test_pad_or_trunc_empty():
    """Test padding from empty array"""
    result = pad_or_trunc([], 5, pad_value=0.0)
    assert len(result) == 5
    assert np.allclose(result, [0.0, 0.0, 0.0, 0.0, 0.0])
    print("✓ Empty array test passed")


def test_pad_or_trunc_none():
    """Test padding from None"""
    result = pad_or_trunc(None, 5, pad_value=1.0)
    assert len(result) == 5
    assert np.allclose(result, [1.0, 1.0, 1.0, 1.0, 1.0])
    print("✓ None input test passed")


def test_pad_or_trunc_short():
    """Test padding short array"""
    result = pad_or_trunc([1, 2], 5, pad_value=0.0)
    assert len(result) == 5
    assert np.allclose(result, [1.0, 2.0, 0.0, 0.0, 0.0])
    print("✓ Short array padding test passed")


def test_pad_or_trunc_exact():
    """Test exact length array"""
    result = pad_or_trunc([1, 2, 3, 4, 5], 5, pad_value=0.0)
    assert len(result) == 5
    assert np.allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0])
    print("✓ Exact length test passed")


def test_pad_or_trunc_long():
    """Test truncating long array"""
    result = pad_or_trunc([1, 2, 3, 4, 5, 6, 7], 5, pad_value=0.0)
    assert len(result) == 5
    assert np.allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0])
    print("✓ Long array truncation test passed")


def test_pad_or_trunc_inf_padding():
    """Test padding with infinity (for TTC arrays)"""
    result = pad_or_trunc([1.0, 2.0], 5, pad_value=np.inf)
    assert len(result) == 5
    assert result[0] == 1.0
    assert result[1] == 2.0
    assert result[2] == np.inf
    assert result[3] == np.inf
    assert result[4] == np.inf
    print("✓ Infinity padding test passed")


if __name__ == "__main__":
    print("Running pad_or_trunc tests...")
    test_pad_or_trunc_empty()
    test_pad_or_trunc_none()
    test_pad_or_trunc_short()
    test_pad_or_trunc_exact()
    test_pad_or_trunc_long()
    test_pad_or_trunc_inf_padding()
    print("\n✅ All pad_or_trunc tests passed!")
