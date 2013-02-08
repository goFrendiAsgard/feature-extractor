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
    new_records = [record for record in records if not (record[len(record)-1] == best_benchmark) or record==records[0] ]
    records = new_records

old_features = old_records[0][0:len(old_records[0])-1]
old_data = old_records[1:len(old_records)]
new_data = []
# zero matrix
for i in xrange(len(old_data)):
    new_data.append([0]*(len(best_phenotypes)+1))
for i in xrange(len(best_phenotypes)):
    new_feature = best_phenotypes[i]    
    projection = get_projection(new_feature, old_features, old_data)
    print ''
    print new_feature, old_features, projection
    for j in xrange(len(projection)):
        new_data[j][i] = projection[j]
print new_data
print len(old_data)
