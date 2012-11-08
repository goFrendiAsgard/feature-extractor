'''
Created on Oct 23, 2012

@author: gofrendi
'''
from Base import GA_Base
from Base import bin_to_dec, randomizer

class GA(GA_Base):
    '''
    Genetics Algorithm class
    '''
    def do_process_individual(self, individual):
        return individual
        
    def do_calculate_fitness(self, individual):
        fitness = {}
        fitness['value'] = bin_to_dec(individual['default'])
        fitness['zero count'] = individual['default'].count('0')
        return fitness
    
    def do_generate_new_individual(self):
        individual = {}
        individual['default'] = ''
        while len(individual['default'])<5:
            rnd = randomizer.randrange(0,2)
            individual['default'] = individual['default'] + str(rnd)
        return individual
    
    def do_crossover(self, individual_1, individual_2):
        rnd = randomizer.randrange(min(len(individual_1['default']),len(individual_2['default'])))
        gene_1 = individual_1['default'][:rnd]
        gene_2 = individual_1['default'][rnd:]
        gene_3 = individual_2['default'][:rnd]
        gene_4 = individual_2['default'][rnd:]
        individual_1['default'] = gene_1 + gene_4
        individual_2['default'] = gene_2 + gene_3
        return individual_1, individual_2
    
    def do_mutation(self, individual):
        rnd = randomizer.randrange(5)
        lst = list(individual['default'])
        if lst[rnd] == '0':
            lst[rnd] = '1'
        else:
            lst[rnd] = '0'
        individual['default'] = ''.join(lst)
        return individual
    
    

if __name__ == '__main__':
    ga = GA()
    ga.population_size = 20
    ga.fitness_measurement = 'max'
    ga.benchmarks = ['value', 'zero count']
    ga.max_epoch=100
    ga.process()
    print(ga.best_individuals(6, benchmark='value', representation='default'))
    print(ga.best_individuals(benchmark='value', representation='default'))
    print(ga.best_individuals(6, benchmark='zero count', representation='default'))
    print(ga.best_individuals(benchmark='zero count', representation='default'))
    for i in xrange(len(ga._individuals)):
        print(ga._individuals[i], ga._fitness[i])
    ga.show()