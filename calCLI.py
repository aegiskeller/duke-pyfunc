#!/usr/bin/env python
from dukelib.calc import add, sub, mul, div
from dukelib.calc import power, sqrt
import click

@click.group()
def cli():
    """A calculator CLI"""

@cli.command("add")
@click.argument("a", type=float)
@click.argument("b", type=float)
def add_cmd(a, b):
    """
    Add two numbers
    example: ./calCLI.py add 2 3
    """
    #use coloured output to print the results
    click.echo(click.style(f"{a} + {b} = {add(a, b)}", fg="green"))

@cli.command("sub") 
@click.argument("a", type=float)
@click.argument("b", type=float)
def sub_cmd(a, b):
    """
    Subtract two numbers
    example:
    ./calCLI.py sub 5 3
    """
    click.echo(click.style(f"{a} - {b} = {sub(a, b)}", fg="red"))

@cli.command("mul")    
@click.argument("a", type=float)
@click.argument("b", type=float)
def mul_cmd(a, b):
    """
    Multiply two numbers
    example:
    ./calCLI.py mul 2 3
    """
    click.echo(click.style(f"{a} * {b} = {mul(a, b)}", fg="blue"))

@cli.command("div")
@click.argument("a", type=float)
@click.argument("b", type=float)
def div_cmd(a, b):
    """
    Divide two numbers
    example:
    ./calCLI.py div 6 3
    """
    click.echo(click.style(f"{a} / {b} = {div(a, b)}", fg="yellow"))

@cli.command("power")
@click.argument("a", type=float)
@click.argument("b", type=float)
def power_cmd(a, b):
    """
    Calculate the power of a number
    example:
    ./calCLI.py power 2 3
    """
    click.echo(click.style(f"{a} ** {b} = {power(a, b)}", fg="magenta"))

@cli.command("sqrt")
@click.argument("a", type=float)
def sqrt_cmd(a):
    """
    Calculate the square root of a number
    example:
    ./calCLI.py sqrt 4
    """
    click.echo(click.style(f"sqrt({a}) = {sqrt(a)}", fg="cyan"))    
    
if __name__ == "__main__":
    cli()
