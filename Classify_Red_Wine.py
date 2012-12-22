from FE_Base import feature_extracting, extract_csv

# make feature extractor
attribute = extract_csv('winequality-red.csv', delimiter=';')
variables = attribute['variables']
records = attribute['data']
feature_extracting(records, variables, label='Red Wine Quality', fold=10)
