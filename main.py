# use the fastapi library to create a simple web server
# to create a logistics web page

from fastapi import FastAPI
from pydantic import BaseModel
from dukelib.logistics import distance, coordinates, total_distance, cities_list

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the logistics API. You can use this API to calculate the distance between two cities in Australia."} 

@app.get("/cities")
async def get_cities():
    """Returns a list of cities for further information
        to get the list of cities we call the cities_list function from the logistics.py file
    """
    cities = cities_list()
    return {"cities": cities}

#build a post request to calculate the distance between two cities
@app.post("/distance")
async def get_distance(city1: str, city2: str):
    """
    This function calculates the distance between two cities in Australia.
    """
    distance_between_cities = distance(city1, city2)
    return {"distance": distance_between_cities}    

# build a post methd to determine the travel time between two cities
# by using the distance between cities and the average speed of 60km/h
@app.post("/travel_time")
async def get_travel_time(city1: str, city2: str):
    """
    This function calculates the travel time between two cities in Australia.
    """
    distance_between_cities = distance(city1, city2)
    travel_time = distance_between_cities / 60
    return {"travel_time": travel_time}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000, host='0.0.0.0')