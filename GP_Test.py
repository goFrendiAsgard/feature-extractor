'''
Created on Nov 11, 2012

@author: gofrendi
'''

from utils import execute
from classes import Genetics_Programming

class GE(Genetics_Programming):
    def do_calculate_fitness(self, individual):
        fitness = {}
        record = [5,4]
        val = 9
        result, error = execute(individual['phenotype'], record)
        if error:
            fitness['default'] = 100000
        else:
            fitness['default'] = abs(val - result)
        return fitness

if __name__ == '__main__':
    gp = GE()
    gp.individual_length = 50
    gp.population_size = 10
    gp.fitness_measurement = 'min'
    gp.nodes = [['x','y'],['plus','minus','multiply','divide']]
    gp.max_epoch=10
    gp.process()
    print(gp.best_individuals(6, representation='default'))
    print(gp.best_individuals(representation='default'))
    print(gp.best_individuals(6, representation='phenotype'))
    print(gp.best_individuals(representation='phenotype'))
    for i in xrange(len(gp._individuals)):
        print(gp._individuals[i], gp._fitness[i])
    gp.show()