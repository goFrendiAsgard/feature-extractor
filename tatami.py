from Feature_Extractor import *

class GE_Tatami(GE_Multi_Accuration_Fitness):
    
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GE_Multi_Accuration_Fitness.__init__(self, records, fold_count, fold_index, classifier)
        self.tatami_best_phenotypes = []
        
    def process(self):
        training_data = self.training_data
        training_label_target = self.training_label_target
        records = list(training_data)
        for i in xrange(len(records)):
            records[i].append(training_label_target[i])
        # add record headers
        record_header = list(self.features)
        record_header.append('class')
        records.insert(0,record_header)
        best_phenotypes = []
        ommited_classes = []
        while(len(ommited_classes) < len(self.benchmarks)-1):
            fe = GE_Local_Separability_Fitness(records, 1, 0)
            fe.max_epoch = self.max_epoch
            fe.process()
            # look for best benchmark in this iteration
            best_fitness = 0
            best_benchmark = ''
            best_phenotype = ''
            for benchmark in fe.benchmarks:
                phenotype = fe.best_individuals(1, benchmark, 'phenotype')
                fitness = fe.best_fitnesses(1, benchmark)
                # look for the best one
                if fitness>best_fitness:
                    best_fitness = fitness
                    best_benchmark = benchmark
                    best_phenotype = phenotype
            # get best_phenotypes, and omitted_classes
            best_phenotypes.append(best_phenotype)
            ommited_classes.append(best_benchmark)
            # remove the class from the record
            new_records = [record for record in records if not (record[len(record)-1] == best_benchmark) or record==records[0] ]
            records = new_records
        # we have best phenotype here, what to do?
        self.tatami_best_phenotypes = best_phenotypes

    def get_new_features(self):
        return self.tatami_best_phenotypes

records = extract_csv('car.csv', delimiter=',')
new_records = []
new_records.append(records[0])
labels = {}
for i in xrange(len(records)):
    if i==0:
        continue
    record = records[i]
    label = record[len(record)-1]
    if not label in labels:
        labels[label] = 0
    if labels[label]>=60:
        continue
    else:
        labels[label] += 1
        new_records.append(record)
        
#records = shuffle_record(records)
records = new_records
extractors = [{'class': GE_Tatami, 'label':'GE_Tatami', 'color':'red', 'params':{}}]

fold_count = 1
extract_feature(records, 'Tatami-whole', fold_count, extractors)

fold_count = 5
extract_feature(records, 'Tatami-5-fold', fold_count, extractors)