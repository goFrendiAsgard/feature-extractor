from Feature_Extractor import *

records = extract_csv('synthesis.csv', delimiter=',')
records = shuffle_record(records)
extractors = None
fold_count = 1

extractors = [
    {'class': GA_Select_Feature, 'label':'GA Select Feature', 'color':'red', 'params':{'max_epoch':100,'population_size':200}},
    {'class': GE_Global_Separability_Fitness, 'label':'GE Global', 'color':'green', 'params':{'max_epoch':100,'population_size':200}},
    {'class': GE_Local_Separability_Fitness, 'label':'GE Local', 'color':'blue', 'params':{'max_epoch':100,'population_size':200}},
    {'class': GE_Multi_Accuration_Fitness, 'label':'GE Multi', 'color':'magenta', 'params':{'max_epoch':100,'population_size':200}},
    {'class': GE_Tatami, 'label':'GE Tatami', 'color':'black', 'params':{'max_epoch':100,'population_size':200}},
]

extract_feature(records, 'Synthesis (whole) - Normal SVM', fold_count, extractors)
fold_count = 5
extract_feature(records, 'Synthesis (5 fold) - Normal SVM', fold_count, extractors)