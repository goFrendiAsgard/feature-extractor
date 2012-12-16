from FE_Base import Feature_Extractor
from gogenpy import utils
    
randomizer = utils.randomizer
records = []
for i in xrange(300):
    x = randomizer.randrange(-7,7)
    y = randomizer.randrange(-7,7)
    r = (x**2+y**2) ** 0.5
    if r<3:
        c = 'kecil'
    elif r<6:
        c = 'sedang'
    else:
        c = 'besar'
    records.append([x,y,c])

variables = ['x','y']


# make feature extractor
fe = Feature_Extractor()
fe.label = 'Hypotenuse'
fe.max_epoch = 200
fe.records = records
fe.population_size = 100
fe.fold = 10
fe.variables = variables
fe.measurement = 'error'
fe.process()