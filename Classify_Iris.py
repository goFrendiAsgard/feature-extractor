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
fe = Feature_Extractor()
fe.label = 'Iris'
fe.max_epoch = 200
fe.records = records
fe.population_size = 100
fe.fold = 5
fe.variables = variables
fe.measurement = 'error'
fe.process()
