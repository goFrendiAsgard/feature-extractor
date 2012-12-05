from FE_Base import Feature_Extractor
from gogenpy import utils
    
randomizer = utils.randomizer
records = []
for i in xrange(240):
    x = randomizer.randrange(-7,7)
    y = randomizer.randrange(-7,7)
    r = (x**2+y**2) ** 0.5
    if r<3:
        c = 0
    elif r<6:
        c = 1
    else:
        c = 2
    records.append([x,y,c])

variables = ['x','y']


# make feature extractor
training_records = records[0:60]
fe = Feature_Extractor()
fe.max_epoch = 200
fe.training_records = training_records
fe.test_records = records
fe.variables = variables
fe.measurement = 'error'
fe.process()

'''
training_records = records[61:120]
fe = Feature_Extractor()
fe.max_epoch = 200
fe.training_records = training_records
fe.test_records = records
fe.variables = variables
fe.measurement = 'error'
fe.process()

training_records = records[121:180]
fe = Feature_Extractor()
fe.max_epoch = 200
fe.training_records = training_records
fe.test_records = records
fe.variables = variables
fe.measurement = 'error'
fe.process()

training_records = records[181:240]
fe = Feature_Extractor()
fe.max_epoch = 200
fe.training_records = training_records
fe.test_records = records
fe.variables = variables
fe.measurement = 'error'
fe.process()
'''