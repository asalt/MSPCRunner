from setuptools import setup, find_packages

version = "0.0.1"

setup(
    name="mspcrunner",
    version=version,
    # packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        #'requests',
        #'importlib; python_version >= "3.6"',
    ],
    packages=["mspcrunner", "mspcrunner.monitor", "mspcrunner.dagster"],
    # packages=find_packages(),
    package_dir={"": "src"},
    author="Alex Saltzman",
    author_email="a.saltzman920@gmail.com",
    description="CLI runner interface for processing mass spec proteomics data",
    entry_points="""
    [console_scripts]
    mspcrunner=mspcrunner.main:app
    """,
    # py_modules = ['main'],
    package_data={
        "mspcrunner": ["../../ext/*", "../../params/*", "./ext/*", "./ext/*/*"],
        "": ["params/*"],
    },
    include_package_data=True,
)
