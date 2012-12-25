from FE_Base import feature_extracting, extract_csv, shuffle_record

# make feature extractor
attribute = extract_csv('winequality-red.csv', delimiter=';')
variables = attribute['variables']
records = attribute['data']
records = shuffle_record(records)

feature_extracting(records, variables, label='Red-Wine-Quality-5-Fold', fold=5)
feature_extracting(records, variables, label='Red-Wine-Quality-1-Fold', fold=1)