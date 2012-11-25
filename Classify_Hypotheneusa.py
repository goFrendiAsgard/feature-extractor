from FE_Base import Feature_Extractor
import utils
    
randomizer = utils.randomizer
records = []
for i in xrange(200):
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
training_records = records[0:60]
variables = ['x','y']


# make feature extractor
fe = Feature_Extractor()
fe.training_records = training_records
fe.test_records = records
fe.variables = variables
fe.measurement = 'error'
fe.process()
