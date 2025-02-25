#!/usr/bin/env python
from dukelib.logistics import distance, coordinates, total_distance
import click

#build a click group
@click.group()
def cli():
    """Tool for working with logistics in Australia"""

# build a click command to calculate the distance between two cities
@cli.command("distance")
@click.argument('city1')
@click.argument('city2')
def distance_cmd(city1, city2):
    """
    This function calculates the distance between two cities in Australia.
    """
    print(distance(city1, city2))

# build a click command to find the coordinates of a city
@cli.command("coordinates")
@click.argument('city')
def coordinates_cmd(city):
    """
    This function returns the coordinates of a city in Australia.
    """
    print(coordinates(city))

# build a click command to calculate the total distance between a list of cities
@cli.command("total-distance")
def total_distance_cmd():
    """
    This function calculates the total distance between a list of cities in Australia.
    """
    print(total_distance())

if __name__ == "__main__":
    cli()