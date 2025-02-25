# test the functions defined in the logistics.py file

from dukelib.logistics import distance, coordinates, total_distance, cities_list
from fastapi.testclient import TestClient
from main import app
import pytest

def test_distance():
    assert distance('Sydney', 'Brisbane') == 730.4061063515427
    assert distance('Sydney', 'Melbourne') == 713.8576651174378 

def test_coordinates():
    assert coordinates('Sydney') == (-33.8688, 151.2093)
    assert coordinates('Melbourne') == (-37.8136, 144.9631)

def test_total_distance():
    assert total_distance() == 11431.57861413497

def test_cities_list():
    assert cities_list() == ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Gold Coast', 'Newcastle', 'Canberra', 'Wollongong', 'Sunshine Coast']

# test the FastAPI web server

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

def test_read_main(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the logistics API. You can use this API to calculate the distance between two cities in Australia."}

def test_read_cities(client):
    response = client.get("/cities")
    assert response.status_code == 200
    assert response.json() == {"cities": ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Gold Coast', 'Newcastle', 'Canberra', 'Wollongong', 'Sunshine Coast']}


#def test_read_distance(client):
#    response = client.post("/distance", json={"city1": "Sydney", "city2": "Brisbane"})
#    assert response.status_code == 200
#    assert response.json() == {"distance": 730.4061063515427}