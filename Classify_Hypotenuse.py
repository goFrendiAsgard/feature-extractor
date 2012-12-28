from Feature_Extractor import *
from gogenpy import utils
    
randomizer = utils.randomizer
records = ['x','y','class']
i = 0
while i<3:
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

records = shuffle_record(records)
fold_count = 3
data_label = 'Hypotenuse'
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
