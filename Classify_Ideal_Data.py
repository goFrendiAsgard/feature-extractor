from Feature_Extractor import *
from gogenpy import utils
    
randomizer = utils.randomizer
records = [['x','y','z','r1','r2','r3','r4','class']]
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

records = shuffle_record(records)
fold_count = 3
data_label = 'Test'
extractors = [
    {'class': GA_Select_Feature, 'label':'GA', 'color':'red'},
    {'class': GP_Select_Feature, 'label':'GP', 'color':'orange'},
    {'class': GP_Global_Separability_Fitness, 'label':'GP Global', 'color':'green'},
    {'class': GP_Local_Separability_Fitness, 'label':'GP Local', 'color':'blue'},
    {'class': GE_Select_Feature, 'label':'GE', 'color':'cyan'},
    {'class': GE_Global_Separability_Fitness, 'label':'GE Global', 'color':'magenta'},
    {'class': GE_Local_Separability_Fitness, 'label':'GE Local', 'color':'black'}
]
extract_feature(records, data_label, fold_count, extractors)
