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
test_phenotype(records, variables,'sqr(abs(sqrt(sepal_width)) / petal_length)')
test_phenotype(records, variables,'sqrt(petal_length * sqr(sepal_width)) - abs(petal_length) / sepal_length * abs(petal_width)')
test_phenotype(records, variables,'sepal_length - sepal_length * sepal_length + sepal_length * petal_width / sepal_width + cos(petal_width / sepal_length) - cos(sepal_width) * sin(petal_width) + petal_length')
test_phenotype(records, variables,'sqr(abs(sqr(petal_width * sqr(sepal_length * sin(sepal_width / sepal_length)) - sepal_width - sqrt(sqr(sqr(cos(sepal_length)) + petal_length - abs(sqrt(cos(petal_width) + sepal_length)) * sepal_width * abs(sqr(sepal_length)) - petal_width / sepal_length)))) - sqr(sin(sepal_width) / petal_length / petal_length) - sin(petal_length + petal_length - abs(sqr(petal_width))))')
'''
feature_extracting(records, variables, label='Iris-5-Fold', fold=5)
feature_extracting(records, variables, label='Iris-1-Fold', fold=1)