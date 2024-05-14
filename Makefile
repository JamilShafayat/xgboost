install:
	sudo apt install python3-pip
	sudo apt install python3-virtualenv
	virtualenv .venv
	source ${PWD}/.venv/bin/activate
build:
	mkdir -p models
	pip3 install -r requirements.txt
run:
	python main.py
