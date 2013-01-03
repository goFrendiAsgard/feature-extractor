from Feature_Extractor import *

fold_count = 1
fold_index = 0
records = extract_csv('iris.data.csv', delimiter=',')
records = shuffle_record(records)
for i in xrange(len(records)):
    class_index = len(records[i])-1
    if not records[i][class_index] == 'Iris-setosa':
        records[i][class_index] = 'other' 

class GE_Madness(GE_Select_Feature):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GE_Select_Feature.__init__(self, records, fold_count, fold_index, classifier)
        self.benchmarks = ['madness']
        self.classifier = svm.SVC(kernel='linear')
    def do_calculate_fitness(self, individual):
        new_feature = individual['phenotype']        
        group_projection = get_projection(new_feature, self.features, self.data, self.training_data, self.training_label_target)
        
        fitness = my_metric(group_projection)
        mean_fitness = 0.0
        for label in fitness:
            mean_fitness += fitness[label]
        mean_fitness /= len(fitness)
        
        accuracy = self.get_accuracy([new_feature])
        madness = mean_fitness /max(float(accuracy['training']), LIMIT_ZERO)
        return {'madness':madness}
    def get_new_features(self):
        best_phenotype = self.best_individuals(1, 'madness', 'phenotype')
        return [best_phenotype]


extractors = [
    {'class': GE_Madness, 'label':'GE Madness', 'color':'black', 
     'params':{'max_epoch':5,'population_size':500, 'new_rate':0.55, 
               'mutation_rate':0.25, 'crossover_rate':0.15, 'elitism_rate':0.05}
    }
]
extract_feature(records, 'test', fold_count, extractors)