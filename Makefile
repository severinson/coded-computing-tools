setup:
	python3 -m venv venv
	venv/bin/pip install -U -r requirements.txt
	echo activate the virtual environment with 'source venv/bin/activate'

test:
	venv/bin/python3 -m unittest tests/*.py

check:
	pylint3 -E ./*
