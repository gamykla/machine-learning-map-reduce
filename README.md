# Machine Learning with MapReduce

Solving some machine learning problems with map-reduce

## What is this?
Many popular machine learning algorithms can be solved with map reduce as discussed in this paper http://papers.nips.cc/paper/3150-map-reduce-for-machine-learning-on-multicore.pdf

Furthermore, it is demonstrated that these common algorithms can scale linearly to the number of processors involved in the computation.

To take advantage of parallelism, algorithms need to be re-formed so that they can be expressed as map reduce operations. This project will demonstrate some of those techniques, starting with simpler problems such as logistic regression with gradient descent expressed as a map reduce computation

### included learning algorithms
* linear regression
* logistic regression


## Running the examples
First install the requirements
```
pip install -r requirements.txt
pip install .
```
Next you'll want to start your 'cluster'. I have 8 cores so i'll setup 8 workers:
```
ipcluster start -n 8
```

Once the cluster has started you can run the python examples:
```
cd mlmapreduce/learning
python linear_regression_mapreduce.py
python logistic_regression_mapreduce.py
```

NB: ipyparallel can run locally, or remotely.. You can build up a cluster to meet your needs, see http://ipyparallel.readthedocs.io/en/latest/ for more detail.

## Running the tests
Tests are implemented with Nose
```
nosetests .
```


## Refernces
* ML datasources: http://archive.ics.uci.edu/ml/datasets.html?area=&att=&format=&numAtt=&numIns=&sort=attup&task=reg&type=&view=table
* http://mran.revolutionanalytics.com/documents/data/
