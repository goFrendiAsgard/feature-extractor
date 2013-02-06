from Feature_Extractor import *
from sklearn import svm

records = extract_csv('iris.data.csv', delimiter=',')
records = shuffle_record(records)
extractors = None
fold_count = 1
'''
extractors = [
    {'class': GP_Multi_Accuration_Fitness, 'label':'GP Multi', 'color':'magenta', 'params':{'max_epoch':100,'population_size':200}},
    {'class': GE_Multi_Accuration_Fitness, 'label':'GE Multi', 'color':'magenta', 'params':{'max_epoch':100,'population_size':200}}
]
'''
extract_feature(records, 'Iris (whole) - Normal SVM coba', fold_count, extractors)
fold_count = 5
extract_feature(records, 'Iris (5 fold) - Normal SVM', fold_count, extractors)