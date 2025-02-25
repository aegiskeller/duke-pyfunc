# test the functions defined in the logistics.py file

from dukelib.logistics import distance, coordinates, total_distance

def test_distance():
    assert distance('Sydney', 'Brisbane') == 730.4061063515427
    assert distance('Sydney', 'Melbourne') == 713.8576651174378 

def test_coordinates():
    assert coordinates('Sydney') == (-33.8688, 151.2093)
    assert coordinates('Melbourne') == (-37.8136, 144.9631)

def test_total_distance():
    assert total_distance() == 11431.57861413497