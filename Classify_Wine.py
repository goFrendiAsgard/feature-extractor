from FE_Base import Feature_Extractor
from wine_data import records, variables
    
training_records = records[0:20]+records[59:79]+records[130:150]


# make feature extractor
fe = Feature_Extractor()
fe.max_epoch = 120
fe.training_records = training_records
fe.test_records = records
fe.variables = variables
fe.measurement = 'error'
fe.process()
