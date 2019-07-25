"""
Unit and regression test for the geometry_analysis package.
"""

# Import package, test suite, and other packages as needed
import geometry_analysis
import pytest
import sys

import numpy as np

@pytest.fixture()   # make 'water_molecule()' an object you can pass to other functions
                    # scope of the decorator can be setted 
def water_molecule():
    name = 'water'
    symbols = ['H', 'O', 'H']
    coordinates = np.array([[2,0,0], [0,0,0], [-2,0,0]])

    water = geometry_analysis.Molecule(name, symbols, coordinates)

    return water

# test error is successfully raised
def test_create_failure():
    name = 25
    symbols = ['H','O','H']
    coordinates = np.zeros([3,3])
    with pytest.raises(TypeError):
        water = geometry_analysis.Molecule(name, symbols, coordinates)

def test_molecule_set_coordinates(water_molecule):
    """Test bond list is rebuilt when we reset coordinates"""
    num_bonds = len(water_molecule.bonds)
    assert num_bonds == 2

    new_coordinates = np.array([[5, 0, 0], [0, 0, 0], [-2, 0, 0]])
    water_molecule.coordinates = new_coordinates
    new_num_bonds = len(water_molecule.bonds)
    assert new_num_bonds == 1
    assert np.array_equal(new_coordinates, water_molecule.coordinates)


def test_geometry_analysis_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "geometry_analysis" in sys.modules

def test_calculate_distance():
    """Test the calculate_distance function"""

    r1 = np.array([0, 0, -1])
    r2 = np.array([0, 1, 0])

    expected_distance = np.sqrt(2.)

    calculated_distance = geometry_analysis.calculate_distance(r1, r2)

    assert expected_distance == calculated_distance

def test_calculate_angle_90():
    """Test the calculate_angle function"""

    rA = np.array([1., 0, 0])
    rB = np.array([0, 0, 0])
    rC = np.array([0, 0, 1.])

    expected_angle_degree = 90.
    calculated_angle_degree = geometry_analysis.calculate_angle(rA, rB, rC, degrees=True)

    np.testing.assert_almost_equal(calculated_angle_degree, expected_angle_degree, decimal=13)
    
def test_calculate_angle_60():
    """Test the calculate_angle function"""

    rA = np.array([1., 0, 0])
    rB = np.array([0, 1., 0])
    rC = np.array([0, 0, 1.])

    expected_angle_degree = 60.
    calculated_angle_degree = geometry_analysis.calculate_angle(rA, rB, rC, degrees=True)

    assert np.isclose(expected_angle_degree, calculated_angle_degree)

# write a decorator for test functions
@pytest.mark.parametrize("p1, p2, p3, expected_angle, tol", [
    (np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([0, 1, 0]), 90, 1e-6),
    (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), 60, 1e-8),
])

def test_calculate_angle(p1, p2, p3, expected_angle, tol):
    calculated_angle = geometry_analysis.calculate_angle(p1, p2, p3, degrees=True)
    assert np.isclose(expected_angle, calculated_angle, rtol=tol)
