# KCluster

This is project is a class that allows for a combination of clustering numeric and string data using kmean, kmedians, and kmodes all in one. You can also weight your variables.

## Installation

Run the following to install: 

'''python
pip install kcluster
'''

## Usage

'''python
from kcluster import KCluster

model = KCluster()
# example that uses kmodes for categorical & kmeans for quantitative
model.fit(X)
'''

## Development kcluster

To install kcluster, along with the tools you need to develop and run tests, run the following in your virtualend:
'''bash
$ pip install -e .[dev]
'''