import pytest
import numpy as np
from modules.compare_saliency_maps import compare_meandif_pxbypx

def test_valid_heatmaps():
    """Test with two valid heatmaps of the same shape."""
    heatmap1 = np.array([[1, 2, 3], [4, 5, 6]])
    heatmap2 = np.array([[1, 1, 1], [1, 1, 1]])
    result = compare_meandif_pxbypx(heatmap1, heatmap2)
    expected = np.mean(np.abs(heatmap1 - heatmap2))
    assert result == pytest.approx(expected), f"Expected {expected}, got {result}"

def test_mismatched_shapes():
    """Test with heatmaps of different shapes."""
    heatmap1 = np.array([[1, 2, 3], [4, 5, 6]])
    heatmap2 = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Heatmaps must have the same shape for comparison"):
        compare_meandif_pxbypx(heatmap1, heatmap2)

def test_with_nans():
    """Test with heatmaps containing NaN values."""
    heatmap1 = np.array([[1, np.nan, 3], [4, 5, 6]]) # 
    heatmap2 = np.array([[1, 1, 1], [1, 1, 1]])
      # resulting diff:   0, nan, 3, 3, 4, 5 -> mean = (0+3+3+4+5)/5 = 3.0
    result = compare_meandif_pxbypx(heatmap1, heatmap2)
    expected = np.nanmean(np.abs(heatmap1 - heatmap2))
    print(f"Testing test_with_nans(). Result: {result}, Expected: {expected}")
    assert result == pytest.approx(expected), f"Expected {expected}, got {result}"

def test_identical_heatmaps():
    """Test with two identical heatmaps."""
    heatmap1 = np.array([[1, 2, 3], [4, 5, 6]])
    heatmap2 = np.array([[1, 2, 3], [4, 5, 6]])
    result = compare_meandif_pxbypx(heatmap1, heatmap2)
    expected = 0.0
    assert result == pytest.approx(expected), f"Expected {expected}, got {result}"

def test_all_nans():
    """Test with heatmaps where all values are NaN."""
    heatmap1 = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    heatmap2 = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    result = compare_meandif_pxbypx(heatmap1, heatmap2)
    assert np.isnan(result), "Expected result to be NaN when all values are NaN"

def test_large_heatmaps():
    """Test with large heatmaps."""
    heatmap1 = np.random.rand(1000, 1000)
    heatmap2 = np.random.rand(1000, 1000)
    result = compare_meandif_pxbypx(heatmap1, heatmap2)
    expected = np.nanmean(np.abs(heatmap1 - heatmap2))
    assert result == pytest.approx(expected), f"Expected {expected}, got {result}"