"""
This module deals with logistics. It claulates the distance between two points 
and the time taken to travel between them and other logists questions.
"""
import geopy.distance   # import the geopy library

#create a list of 10 cities in Australia and their coordinates

cities = {
    'Sydney': (-33.8688, 151.2093),
    'Melbourne': (-37.8136, 144.9631),
    'Brisbane': (-27.4698, 153.0251),
    'Perth': (-31.9505, 115.8605),
    'Adelaide': (-34.9285, 138.6007),
    'Gold Coast': (-28.0167, 153.4000),
    'Newcastle': (-32.9283, 151.7817),
    'Canberra': (-35.2809, 149.1300),
    'Wollongong': (-34.4240, 150.8931),
    'Sunshine Coast': (-26.6500, 153.0667)
}   

# build a function to claualte the distance between two cities

def distance(city1, city2):
    """
    This function calculates the distance between two cities in Australia.
    """
    # get the coordinates of the two cities
    coord1 = cities[city1]
    coord2 = cities[city2]  
    # calculate the distance between the two cities
    # distance('Sydney', 'Brisbane')
    # Out[2]: 730.4061063515427
    distance = geopy.distance.distance(coord1, coord2).km
    return distance

# build a function that finds the coordinates of a city

def coordinates(city):
    """
    This function returns the coordinates of a city in Australia.
    In [3]: coordinates('Sydney')
    Out[3]: (-33.8688, 151.2093)

    """
    return cities[city] # return the coordinates of the city    

# calaulate the total distance between a list of cities

def total_distance():
    """
    This function calculates the total distance between a list of cities in Australia.
    In [2]: total_distance()
    Out[2]: 11431.57861413497
    """
    total_distance = 0
    for i in range(len(cities)-1):
        total_distance += distance(list(cities.keys())[i], list(cities.keys())[i+1])
    return total_distance   




