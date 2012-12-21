from FE_Base import extract_csv

attribute = extract_csv('winequality-red', delimiter=';')
variables = attribute['variables']
data = attribute['data']