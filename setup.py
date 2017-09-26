# Necessary to supress on error in Python 2.7.3 at the completion of
# python setup.py test.
# See http://bugs.python.org/issue15881#msg170215
import multiprocessing        # NOQA

from setuptools import find_packages
import distutils.command.clean
import os
import setuptools
import subprocess


class Clean(distutils.command.clean.clean):
    def run(self):
        subprocess.call('find . -name *.pyc -delete'.split(' '))
        distutils.command.clean.clean.run(self)


def read_file(file_name):
    """Utility function to read a file."""
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


def read_first_line(file_name):
    """Read the first line from the specified file."""
    with open(os.path.join(os.path.dirname(__file__), file_name)) as f:
        return f.readline().strip()


def read_requirements(file_path):
    return [
        i
        for i in [j.strip() for j in open(file_path).readlines()]
        if i and not i.startswith(('#', '-'))
    ]


REQUIREMENTS = read_requirements('requirements/requirements.txt')
TEST_REQUIREMENTS = read_requirements('requirements/requirements.txt')

setuptools.setup(name='mlmapreduce',
                 version='1.0,0',
                 description="mlmapreduce machine learning stuff",
                 long_description=read_file('README.md'),
                 classifiers=[
                      'Development Status :: 3 - Alpha',
                      'Environment :: Web Environment',
                      'Intended Audience :: Developers',
                      'License :: Other/Proprietary License',
                      'Natural Language :: English',
                      'Operating System :: POSIX :: Linux',
                      'Programming Language :: Python',
                      'Programming Language :: Python :: 2.7',
                      'Topic :: Internet :: WWW/HTTP :: Dynamic Content :: CGI Tools/Libraries',
                      'Topic :: Software Development :: Libraries :: Python Modules'
                 ],
                 keywords='machine learning',
                 author='gamykla',
                 author_email='gamykla@github',
                 url='',
                 license='',
                 packages=find_packages(exclude=['tests']),
                 include_package_data=True,
                 zip_safe=False,
                 install_requires=REQUIREMENTS,
                 entry_points="""
                 # -*- Entry points: -*-
                 """,
                 test_suite='nose.collector',
                 tests_require=TEST_REQUIREMENTS,
                 cmdclass={
                     'clean': Clean
                 }
                 )