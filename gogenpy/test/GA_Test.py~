'''
Created on Nov 8, 2012

@author: gofrendi
'''

from utils import bin_to_dec
from classes import Genetics_Algorithm

class GA(Genetics_Algorithm):
    '''
    GA class
    '''
            
    def do_calculate_fitness(self, individual):
        fitness = {}
        fitness['value'] = bin_to_dec(individual['default'])
        fitness['zero count'] = individual['default'].count('0')
        return fitness   
    

if __name__ == '__main__':
    ga = GA()
    ga.individual_length = 20
    ga.population_size = 10
    ga.fitness_measurement = 'max'
    ga.benchmarks = ['value', 'zero count']
    ga.max_epoch=100
    ga.stopping_value = {'value':1024000, 'zero count': 16}
    ga.process()
    '''
    print(ga.best_individuals(6, benchmark='value', representation='default'))
    print(ga.best_individuals(benchmark='value', representation='default'))
    print(ga.best_individuals(6, benchmark='zero count', representation='default'))
    print(ga.best_individuals(benchmark='zero count', representation='default'))
    for i in xrange(len(ga._individuals)):
        print(ga._individuals[i], ga._fitness[i])
    '''
    ga.show()
