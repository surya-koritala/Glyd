# The package carries a platform binary (libglyd_store), so its wheel is
# tagged for the platform it was built on, not "any".
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
