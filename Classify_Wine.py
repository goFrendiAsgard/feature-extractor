from FE_Base import feature_extracting, extract_csv

# make feature extractor
attribute = extract_csv('wine.data.csv', delimiter=',')
variables = attribute['variables']
records = attribute['data']
feature_extracting(records, variables, label='Wine Data', fold=1)
