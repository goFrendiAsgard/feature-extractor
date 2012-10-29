'''
Created on Oct 23, 2012

@author: gofrendi
'''
from Base import GA_Base

class GA(GA_Base):
    '''
    Genetics Algorithm class
    '''
    def do_process_individual(self, individual):
        return GA_Base.do_process_individual(self, individual)
        
    def do_calculate_fitness(self, individual):
        return GA_Base.do_calculate_fitness(self, individual)
    
    def do_generate_new_individual(self):
        return GA_Base.do_generate_new_individual(self)
    
    def do_crossover(self, individual_1, individual_2):
        return GA_Base.do_crossover(self, individual_1, individual_2)
    
    def do_mutation(self, individual):
        return GA_Base.do_mutation(self, individual)
    
    

if __name__ == '__main__':
    ga = GA()
    ga.process()
    ga.show()