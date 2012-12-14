from FE_Base import Feature_Extractor
from gogenpy import utils
    
randomizer = utils.randomizer
records = []
for i in xrange(100):
    c = 'class 1';
    x = 0
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        y = randomizer.randrange(1,5)
    else:
        y = randomizer.randrange(-4,0)
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        z = randomizer.randrange(1,5)
    else:
        z = randomizer.randrange(-4,0)
    records.append([x,y,z,c])

for i in xrange(100):
    c = 'class 2';
    y = 0
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        x = randomizer.randrange(1,5)
    else:
        x = randomizer.randrange(-4,0)
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        z = randomizer.randrange(1,5)
    else:
        z = randomizer.randrange(-4,0)
    records.append([x,y,z,c])

for i in xrange(100):
    c = 'class 3';
    z = 0
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        y = randomizer.randrange(1,5)
    else:
        y = randomizer.randrange(-4,0)
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        x = randomizer.randrange(1,5)
    else:
        x = randomizer.randrange(-4,0)
    records.append([x,y,z,c])

variables = ['x','y','z']


# make feature extractor
fe = Feature_Extractor()
fe.label = 'Ideal data'
fe.max_epoch = 200
fe.records = records
fe.population_size = 100
fe.fold = 5
fe.variables = variables
fe.measurement = 'error'
fe.process()