from Feature_Extractor import *

records = extract_csv('wine.data.csv', delimiter=',')
records = shuffle_record(records)
fold_count = 5
data_label = 'Wine Data'
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