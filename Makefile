test:
	python3 -m unittest tests/*.py

check:
	pylint3 -E ./*
