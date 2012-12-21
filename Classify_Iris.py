from FE_Base import feature_extracting
from sklearn import datasets
    
ds = datasets.load_iris()
data = list(ds.data)
target = list(ds.target)
records = []
for i in xrange(len(data)):
    record = []
    for j in xrange(len(data[i])):
        record.append(data[i][j])
    record.append(target[i])
    records.append(record)
variables = ['petal_length','petal_width','sepal_length','sepal_width']


# make feature extractor
feature_extracting(records, variables, label='Iris', fold=1)