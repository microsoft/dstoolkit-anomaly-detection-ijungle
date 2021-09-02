
VERSION := $(shell echo `head -1 src/iJungle/config.py | gawk -F"'" '{print $$2}'`)

dist/iJungle-$(VERSION)-py3-none-any.whl:
	python setup.py bdist_wheel

install: dist/iJungle-$(VERSION)-py3-none-any.whl
	pip install dist/iJungle-$(VERSION)-py3-none-any.whl

clean:
	rm -rf build
	rm -rf dist
	rm -rf iJungle.egg-info

all: clean install
