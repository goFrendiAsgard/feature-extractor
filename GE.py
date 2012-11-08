'''
Created on Nov 7, 2012

@author: gofrendi
'''

from Base import GA_Base

class Grammatical_Evolution(GA_Base):
    
    def __init__(self):
        super(Grammatical_Evolution, self).__init__()
        self.__grammar = None
    
    def do_calculate_fitness(self, individual):
        return GA_Base.do_calculate_fitness(self, individual)
    
    def do_crossover(self, individual_1, individual_2):
        return GA_Base.do_crossover(self, individual_1, individual_2)
    
    def do_mutation(self, individual):
        return GA_Base.do_mutation(self, individual)
    
    def do_generate_new_individual(self):
        return GA_Base.do_generate_new_individual(self)
    
    def do_process_individual(self, individual):
        return GA_Base.do_process_individual(self, individual)
