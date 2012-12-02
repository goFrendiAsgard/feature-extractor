from FE_Base import Feature_Extractor
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
training_records = records[0:20]+records[50:70]+records[130:150]
fe = Feature_Extractor()
fe.max_epoch = 100
fe.training_records = training_records
fe.test_records = records
fe.variables = variables
fe.measurement = 'error'
fe.process()
