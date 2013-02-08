from Feature_Extractor import *

records = extract_csv('iris.data.csv', delimiter=',')
records = shuffle_record(records)
old_records = list(records)
fold_count = 1
best_phenotypes = []
ommited_classes = []

while(len(records)>1):
    fe = GE_Multi_Accuration_Fitness(records, fold_count, 0)
    fe.max_epoch=50
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
    new_records = [record for record in records if not (record[len(record)-1] == best_benchmark) ]
    records = new_records

print best_phenotypes
old_data = old_records[1:len(old_records)]
# get_projection(new_feature, old_features, all_data, used_data = None, used_target = None):
print len(old_data)
