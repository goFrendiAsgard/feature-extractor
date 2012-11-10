'''
Created on Nov 9, 2012

@author: gofrendi
'''

from utils import execute
from classes import Grammatical_Evolution

class GE(Grammatical_Evolution):
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
    ge = GE()
    ge.individual_length = 50
    ge.population_size = 10
    ge.fitness_measurement = 'min'
    ge.variables = ['x','y']
    ge.start_node = 'expr'
    ge.grammar={
        'expr':['var','expr op expr'],
        'var' :['x','y'],
        'op'  :['+','-','*','/']
    }    
    ge.max_epoch=10
    ge.process()
    print(ge.best_individuals(6, representation='default'))
    print(ge.best_individuals(representation='default'))
    print(ge.best_individuals(6, representation='phenotype'))
    print(ge.best_individuals(representation='phenotype'))
    for i in xrange(len(ge._individuals)):
        print(ge._individuals[i], ge._fitness[i])
    ge.show()