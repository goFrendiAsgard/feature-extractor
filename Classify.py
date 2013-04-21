
import Feature_Extractor as FE
import sys

def extract_feature(csv_file_name):
    records = FE.extract_csv(csv_file_name, delimiter=',')
    records = FE.shuffle_record(records)
    fold_count = 1
    # define extractors
    extractors = [
        {'class': FE.GA_Select_Feature, 'label':'GA Select Feature', 'color':'red', 'params':{'max_epoch':100,'population_size':200}},
        {'class': FE.GE_Global_Separability_Fitness, 'label':'GE Global', 'color':'green', 'params':{'max_epoch':100,'population_size':200}},
        {'class': FE.GE_Multi_Accuration_Fitness, 'label':'GE Multi', 'color':'magenta', 'params':{'max_epoch':100,'population_size':200}},
        {'class': FE.GE_Tatami_Multi_Accuration_Fitness, 'label':'GE Tatami Multi', 'color':'black', 'params':{'max_epoch':100,'population_size':200}},
    ]
    # get label
    file_name_partials = csv_file_name.split('.')
    if(len(file_name_partials)>1):
        label = '.'.join(file_name_partials[0:len(file_name_partials)-1])
    else:
        label = csv_file_name
    # extract feature
    FE.extract_feature(records, label+' (whole)', fold_count, extractors)
    fold_count = 5
    FE.extract_feature(records, label+' (5 fold)', fold_count, extractors)

if __name__ == '__main__':
    if len(sys.argv)>1:
        csv_file_name = sys.argv[1]
    else:
        print('Give me a csv file name:')
        csv_file_name = raw_input()
    extract_feature(csv_file_name)
