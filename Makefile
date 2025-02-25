install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv test_project.py test_logistics.py main.py

format:
	black *.py dukelib/*.py

lint:
	pylint --disable=R,C *.py dukelib/*.py

refactor: lint format

all: install lint testmake test