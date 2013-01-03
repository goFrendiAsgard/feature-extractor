from Feature_Extractor import *
from sklearn import svm

records = extract_csv('iris.data.csv', delimiter=',')
records = shuffle_record(records)
fold_count = 1
'''
extractors = [
    {'class': GA_Select_Feature, 'label':'GA', 'color':'red', 'params':{}},
    {'class': GP_Select_Feature, 'label':'GP', 'color':'orange', 'params':{'max_epoch':100,'population_size':200}},
    {'class': GP_Global_Separability_Fitness, 'label':'GP Global', 'color':'green', 'params':{'max_epoch':100,'population_size':200}},
    {'class': GP_Local_Separability_Fitness, 'label':'GP Local', 'color':'blue', 'params':{'max_epoch':100,'population_size':200}},
    {'class': GE_Select_Feature, 'label':'GE', 'color':'cyan', 'params':{'max_epoch':100,'population_size':200}},
    {'class': GE_Global_Separability_Fitness, 'label':'GE Global', 'color':'magenta', 'params':{'max_epoch':100,'population_size':200}},
    {'class': GE_Local_Separability_Fitness, 'label':'GE Local', 'color':'black', 'params':{'max_epoch':100,'population_size':200}}
]
'''
extractors = None
extract_feature(records, 'Iris - Normal SVM', fold_count, extractors)
extract_feature(records, 'Iris - Linear SVM', fold_count, extractors, svm.SVC(kernel='linear'))