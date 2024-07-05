default:
	cat Makefile

.venv/bin/activate: requirements.txt
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

create: .venv/bin/activate
	. .venv/bin/activate

clean:
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -r {} +
