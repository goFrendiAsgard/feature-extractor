from FE_Base import feature_extracting, shuffle_record, test_phenotype
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
records = shuffle_record(records)

'''
test_phenotype(records, variables, 'sepal_width - petal_width * sepal_length - petal_width * sqr(sepal_width + sqr(petal_length) * petal_length * sqr(sqrt(sqr(sqr(petal_length)) / sepal_length) / sqrt(petal_length + petal_length - sepal_length) * sqr(sqr(sepal_length)))) - sepal_length')
'''

feature_extracting(records, variables, label='Iris-10-Fold', fold=10)
feature_extracting(records, variables, label='Iris-5-Fold', fold=5)
feature_extracting(records, variables, label='Iris-1-Fold', fold=1)