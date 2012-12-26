from FE_Base import feature_extracting, extract_csv, shuffle_record, test_phenotype

# make feature extractor
attribute = extract_csv('glass.data.csv', delimiter=',')
variables = attribute['variables']
records = attribute['data']
records = shuffle_record(records)

feature_extracting(records, variables, label='Glass-5-Fold', fold=5)
feature_extracting(records, variables, label='Glass-3-Fold', fold=3)
feature_extracting(records, variables, label='Glass-1-Fold', fold=1)