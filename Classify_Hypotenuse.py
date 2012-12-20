from FE_Base import Feature_Extractor
from gogenpy import utils
    
randomizer = utils.randomizer
records = []
i = 0
while i<10:
    for x in xrange(-10,10):
        for y in xrange(-10,10):
            r = (x**2+y**2) ** 0.5
            if r<6.4:
                c = 'kecil'
            elif r<9.2:
                c = 'sedang'
            else:
                c = 'besar'
            records.append([x,y,c])
    i+=1

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