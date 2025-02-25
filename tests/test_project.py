# test the functions in the duke-pyfunc calc.py file
from dukelib.calc import add, sub, mul, div, power, sqrt
from calCLI import cli
#import the runner from click to test the CLI
from click.testing import CliRunner

#write a test for each cmd of the CLI
def test_add_cmd():
    #invoke the runner
    runner = CliRunner()
    #run the command
    result = runner.invoke(cli, ["add", "2", "3"])
    #assert the expected output
    assert "2.0 + 3.0 = 5.0\n" in result.output
    assert result.exit_code == 0

def test_sub_cmd():
    runner = CliRunner()
    result = runner.invoke(cli, ["sub", "5", "3"])
    assert result.output == "5.0 - 3.0 = 2.0\n"
    assert result.exit_code == 0

def test_mul_cmd():
    runner = CliRunner()
    result = runner.invoke(cli, ["mul", "2", "3"])
    assert "2.0 * 3.0 = 6.0\n" in result.output
    assert result.exit_code == 0

def test_div_cmd():
    runner = CliRunner()
    result = runner.invoke(cli, ["div", "6", "3"])
    assert "6.0 / 3.0 = 2.0\n" in result.output
    assert result.exit_code == 0

def test_power_cmd():
    runner = CliRunner()
    result = runner.invoke(cli, ["power", "2", "3"])
    assert "8" in result.output
    assert result.exit_code == 0

def test_sqrt_cmd():
    runner = CliRunner()
    result = runner.invoke(cli, ["sqrt", "4"])
    assert "2" in result.output
    assert result.exit_code == 0    

def test_add():
    assert add(2, 3) == 5
    assert add(1, 1) == 2
def test_sub():
    assert sub(5, 3) == 2
    assert sub(2, 1) == 1

def test_mul():
    assert mul(2, 3) == 6
    assert mul(2, 2) == 4   

def test_div():
    assert div(6, 3) == 2
    assert div(4, 2) == 2   

def test_power():
    assert power(2, 3) == 8
    assert power(2, 2) == 4 

def test_sqrt():
    assert sqrt(4) == 2             


