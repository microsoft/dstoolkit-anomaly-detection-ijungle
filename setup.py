from setuptools import setup

with open('src/iJungle/config.py') as config_file:
    exec(config_file.read())

setup(
    name = 'iJungle',
    packages = ['iJungle'],
    package_dir = {'iJungle': 'src/iJungle'},
    version = __version__,
    description = 'Isolation jungle',
    install_requires = ['scikit-learn']
)
