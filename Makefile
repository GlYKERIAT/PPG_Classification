default:
	cat Makefile

.venv/bin/activate: requirements.txt
	python3 -m venv .venv

run: .venv/bin/activate
	. ./.venv/bin/activate

clean:
	rm -rf .venv
	find . -type d -name __pycahce__ -exec -rm -r {} \;
