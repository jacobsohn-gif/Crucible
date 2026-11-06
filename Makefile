.PHONY: all build test install clean

PYTHON := .venv/bin/python

all: test build

build:
	$(PYTHON) scripts/build_release.py

test:
	$(PYTHON) -W error::ResourceWarning -m unittest discover -s tests -v

install: build
	mkdir -p $(HOME)/.local/bin
	ln -sf $(CURDIR)/bin/crux $(HOME)/.local/bin/crux
	ln -sf $(CURDIR)/bin/ush $(HOME)/.local/bin/ush

clean:
	rm -rf build dist *.egg-info
