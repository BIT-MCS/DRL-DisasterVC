from setuptools import setup, find_packages
from mcs.globals import VERSION

# https://github.com/kennethreitz/setup.py/blob/master/setup.py


with open("README.md", "r") as fh:
    long_description = fh.read()

extras = {
    "profiler": ["pyinstrument>=2.0"],
}
test_deps = ["pytest"]

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
all_deps = all_deps + test_deps
extras["all"] = all_deps


setup(
    name="mcs",
    version=VERSION,
    python_requires=">=3.5.0",
    packages=find_packages(),
)
