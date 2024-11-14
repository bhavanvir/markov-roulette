# set default python to python3 if not specified on command line
PYTHON ?= python3

# to use a specific Python version, use `PYTHON=pythonX.XX make dev-setup`.
# Using the default Mac OS X Python can cause warnings from urllib3 and may
# have other impact.  See https://github.com/urllib3/urllib3/issues/3020
dev-setup:
	@$(PYTHON) -m venv venv
	@sh -c '. venv/bin/activate ; echo Installing requirements in venv created at $$VIRTUAL_ENV ; pip install -r requirements.txt'
	@echo "venv created; to activate use '. venv/bin/activate'"
