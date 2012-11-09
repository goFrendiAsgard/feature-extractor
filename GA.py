'''
Created on Oct 23, 2012

@author: gofrendi
'''
from Base import GA_Base
from Base import randomizer

class Genetics_Algorithm(GA_Base):
    '''
    Genetics Algorithm class
    '''
    
    def __init__(self):
        super(Genetics_Algorithm, self).__init__()
        self._individual_length = 5
        
    def do_process_individual(self, individual):
        return individual
    
    def do_generate_new_individual(self):
        individual = {}
        individual['default'] = ''
        while len(individual['default'])<self._individual_length:
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
        rnd = randomizer.randrange(self._individual_length)
        lst = list(individual['default'])
        if lst[rnd] == '0':
            lst[rnd] = '1'
        else:
            lst[rnd] = '0'
        individual['default'] = ''.join(lst)
        return individual
    
    @property
    def individual_length(self):
        return self._individual_length
    @individual_length.setter
    def individual_length(self,value):
        self._individual_length = value