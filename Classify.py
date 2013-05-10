import Feature_Extractor as FE
import sys
from sklearn.tree import DecisionTreeClassifier

def extract_feature(csv_file_name, label = None):
    records = FE.extract_csv(csv_file_name, delimiter=',')
    records = FE.shuffle_record(records)
    groups = []
    for record in records[1:]:
        if record[-1] not in groups:
            groups.append(record[-1])
    group_count = len(groups)
    # define classifier    
    classifier = DecisionTreeClassifier(max_depth=group_count-1, random_state=0)
    # define extractors
    params = {'max_epoch':100,'population_size':100, 'mutation_rate':0.25, 'new_rate':0.5, 
              'elitism_rate':0.05, 'crossover_rate': 0.2, 'stopping_value':1.0}
    extractors = [
        {'class': FE.GA_Select_Feature, 'label':'GA Select Feature', 'color':'red', 'params':params},        
        {'class': FE.GE_Global_Separability_Fitness, 'label':'GE Global', 'color':'blue', 'params':params},
        {'class': FE.GE_Multi_Accuration_Fitness, 'label':'GE Multi', 'color':'cyan', 'params':params},
        {'class': FE.GE_Tatami_Multi_Accuration_Fitness, 'label':'GE Tatami Multi', 'color':'magenta', 'params':params},
        {'class': FE.GE_Gravalis, 'label':'GE Gravalis', 'color':'green','params':params},
    ]
    # get label
    if label is None:
        file_name_partials = csv_file_name.split('.')
        if(len(file_name_partials)>1):
            label = '.'.join(file_name_partials[0:len(file_name_partials)-1])
        else:
            label = csv_file_name
    # extract feature
    fold_count = 1
    FE.extract_feature(records, label+' (whole)', fold_count, extractors, classifier)
    fold_count = 5
    FE.extract_feature(records, label+' (5 fold)', fold_count, extractors, classifier)

if __name__ == '__main__':
    if len(sys.argv)>1:
        csv_file_name = sys.argv[1]
    else:
        print('Give me a csv file name:')
        csv_file_name = raw_input()
    if len(sys.argv)>2:
        label = sys.argv[2]
    else:
        label = None
    extract_feature(csv_file_name, label)
