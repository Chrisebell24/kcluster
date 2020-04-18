from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    author='Christopher Bell',
    author_email='Chris.E.Bell24@gmail.com',
    maintainer='Christopher Bell',
    maintainer_email='Chris.E.Bell24@gmail.com',
    url='https://github.com/chrisebell24/kcluster',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
    ],
    name='kcluster',
    version='0.0.1',
    description='Cluster using a combined kmeans, kmedians, and kmodes. Allows weightings',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=['kcluster', '_util_cluster', '_util_kcluster'],
    package_dir={'': 'src'},
    install_requires = [
        "pandas>=0.25.1",
        "numpy>=1.6.5",
        "scikit-learn>=0.21.3",
    ],
    extras_require = {
        "dev": [
            "pytest >= 3.7",
        ],
    },
)