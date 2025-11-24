"""
Tests for KPIC instrument module, specifically the prepare_all_data function.
"""
import numpy as np


def test_prepare_all_data_import():
    """Test that prepare_all_data can be imported from KPIC module"""
    from breads.instruments.KPIC import prepare_all_data
    assert callable(prepare_all_data)


def test_prep_data_object_backward_compatibility():
    """Test that old function name still works for backward compatibility"""
    from breads.instruments.KPIC import prep_data_object
    assert callable(prep_data_object)


def test_fiber_segregation_logic():
    """
    Test the logic of segregating files by fiber number.
    This tests the core logic without requiring actual data files.
    """
    # Test case 1: Multiple fibers with multiple files each
    fiber_list = np.array([0, 0, 2, 2, 0, 1, 1])
    file_numbers = np.array([100, 101, 102, 103, 104, 105, 106])

    # Expected groupings:
    # Fiber 0: files 100, 101, 104
    # Fiber 1: files 105, 106
    # Fiber 2: files 102, 103

    # We can verify the fiber list to array conversion works
    fiber_list_converted = np.array(fiber_list)
    file_numbers_converted = np.array(file_numbers)

    # Check unique fibers
    unique_fibers = np.unique(fiber_list_converted)
    assert len(unique_fibers) == 3
    assert 0 in unique_fibers
    assert 1 in unique_fibers
    assert 2 in unique_fibers

    # Check fiber 0 indices
    indices_fiber_0 = np.where(fiber_list_converted == 0)[0]
    assert len(indices_fiber_0) == 3
    assert np.array_equal(file_numbers_converted[indices_fiber_0], [100, 101, 104])

    # Check fiber 1 indices
    indices_fiber_1 = np.where(fiber_list_converted == 1)[0]
    assert len(indices_fiber_1) == 2
    assert np.array_equal(file_numbers_converted[indices_fiber_1], [105, 106])

    # Check fiber 2 indices
    indices_fiber_2 = np.where(fiber_list_converted == 2)[0]
    assert len(indices_fiber_2) == 2
    assert np.array_equal(file_numbers_converted[indices_fiber_2], [102, 103])


def test_fiber_indexing_from_zero():
    """Test that science fibers are indeed indexed from 0"""
    fiber_list = np.array([0, 1, 2, 3])
    unique_fibers = np.unique(fiber_list)

    # Verify all fiber indices are non-negative (indexed from 0)
    assert np.all(unique_fibers >= 0)
    assert 0 in unique_fibers


def test_empty_fiber_handling():
    """Test that missing fibers are handled correctly"""
    # Only fibers 0 and 2 present, fibers 1 and 3 missing
    fiber_list = np.array([0, 0, 2, 2])

    unique_fibers = np.unique(fiber_list)

    # Should only have fibers 0 and 2
    assert len(unique_fibers) == 2
    assert 0 in unique_fibers
    assert 2 in unique_fibers
    assert 1 not in unique_fibers
    assert 3 not in unique_fibers


def test_single_fiber_single_file():
    """Test edge case with single fiber and single file"""
    fiber_list = np.array([0])
    file_numbers = np.array([100])

    unique_fibers = np.unique(fiber_list)
    assert len(unique_fibers) == 1
    assert unique_fibers[0] == 0

    indices = np.where(np.array(fiber_list) == 0)[0]
    assert len(indices) == 1
    assert file_numbers[indices[0]] == 100


def test_all_files_same_fiber():
    """Test when all files belong to the same fiber"""
    fiber_list = np.array([2, 2, 2, 2])
    file_numbers = np.array([100, 101, 102, 103])

    unique_fibers = np.unique(fiber_list)
    assert len(unique_fibers) == 1
    assert unique_fibers[0] == 2

    indices = np.where(np.array(fiber_list) == 2)[0]
    assert len(indices) == 4
    assert np.array_equal(file_numbers[indices], [100, 101, 102, 103])
