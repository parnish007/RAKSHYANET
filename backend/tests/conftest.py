"""
Pytest configuration and shared fixtures for RakshyaNet tests
"""
import pytest


@pytest.fixture
def depot_location():
    """Kathmandu Central depot coordinates"""
    return (27.7172, 85.3240)


@pytest.fixture
def sample_village_data():
    return {
        "id": "dhulikhel",
        "name": "Dhulikhel",
        "lat": 27.6200,
        "lng": 85.5500,
        "population": 5000,
        "initial_urgency": 0.65,
        "disaster_impact": 0.70,
        "current_need": 2500.0,
        "min_need": 1500.0,
        "terrain_difficulty": 1.8
    }
