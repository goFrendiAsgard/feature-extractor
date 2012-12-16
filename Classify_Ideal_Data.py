from FE_Base import Feature_Extractor
from gogenpy import utils
    
randomizer = utils.randomizer
records = []
for i in xrange(100):
    c = 'class 1';
    x = 0
    y = randomizer.randrange(1,5)
    z = randomizer.randrange(1,5)
    r1 = randomizer.randrange(1,5)
    r2 = randomizer.randrange(1,5)
    r3 = randomizer.randrange(1,5)
    r4 = randomizer.randrange(1,5)
    records.append([x,y,z,r1,r2,r3,r4,c])

for i in xrange(100):
    c = 'class 2';
    y = 0
    x = randomizer.randrange(1,5)
    z = randomizer.randrange(1,5)
    r1 = randomizer.randrange(1,5)
    r2 = randomizer.randrange(1,5)
    r3 = randomizer.randrange(1,5)
    r4 = randomizer.randrange(1,5)
    records.append([x,y,z,r1,r2,r3,r4,c])

for i in xrange(100):
    c = 'class 3';
    z = 0
    y = randomizer.randrange(1,5)
    x = randomizer.randrange(1,5)
    r1 = randomizer.randrange(1,5)
    r2 = randomizer.randrange(1,5)
    r3 = randomizer.randrange(1,5)
    r4 = randomizer.randrange(1,5)
    records.append([x,y,z,r1,r2,r3,r4,c])

variables = ['x','y','z','r1','r2','r3','r4']


# make feature extractor
fe = Feature_Extractor()
fe.label = 'Ideal data'
fe.max_epoch = 200
fe.records = records
fe.population_size = 100
fe.fold = 10
fe.variables = variables
fe.measurement = 'error'
fe.process()