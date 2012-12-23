from FE_Base import feature_extracting, extract_csv, shuffle_record, test_phenotype

# make feature extractor
attribute = extract_csv('iris.data.csv', delimiter=',')
variables = attribute['variables']
records = attribute['data']
records = shuffle_record(records)

'''
test_phenotype(records, variables,'sepal_width')
test_phenotype(records, variables,'sepal_length')
test_phenotype(records, variables,'petal_width')
test_phenotype(records, variables,'petal_length')
test_phenotype(records, variables, 'sepal_width - petal_width * sepal_length - petal_width * sqr(sepal_width + sqr(petal_length) * petal_length * sqr(sqrt(sqr(sqr(petal_length)) / sepal_length) / sqrt(petal_length + petal_length - sepal_length) * sqr(sqr(sepal_length)))) - sepal_length')
'''
feature_extracting(records, variables, label='Iris-10-Fold', fold=10)
feature_extracting(records, variables, label='Iris-5-Fold', fold=5)
feature_extracting(records, variables, label='Iris-1-Fold', fold=1)
