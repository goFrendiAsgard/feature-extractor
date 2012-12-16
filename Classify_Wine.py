from FE_Base import Feature_Extractor
from wine_data import records, variables
    
training_records = records[0:20]+records[59:79]+records[130:150]


# make feature extractor
fe = Feature_Extractor()
fe.label = 'Wine'
fe.max_epoch = 200
fe.records = records
fe.population_size = 100
fe.fold = 10
fe.variables = variables
fe.measurement = 'error'
fe.process()