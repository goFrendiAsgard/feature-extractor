from FE_Base import feature_extracting, extract_csv, shuffle_record

# make feature extractor
attribute = extract_csv('winequality-white.csv', delimiter=';')
variables = attribute['variables']
records = attribute['data']
records = shuffle_record(records)

feature_extracting(records, variables, label='White-Wine-Quality-5-Fold', fold=5)
feature_extracting(records, variables, label='White-Wine-Quality-1-Fold', fold=1)
