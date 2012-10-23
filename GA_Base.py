'''
Created on Oct 23, 2012

@author: gofrendi
'''

import matplotlib.pyplot as plt
import numpy as np

def bin_to_dec(binary):
    '''
    decimal form of binary
    '''
    return int(binary,2)

def dec_to_bin(decimal):
    '''
    binary form of decimal
    '''
    return str(bin(decimal)[2:])

def bin_digit_needed(decimal):
    '''
    binary digit needed to represent decimal number
    '''
    return len(dec_to_bin(decimal))

    

class GA_Base(object):
    '''
    GA_Base is a class that will be used as super class
    of every Genetics Algorithm based.
    This class is not intended for direct usage.
    
    If you want to make any inheritance of this class, there are view things to be considered
       
    
    
    HOW TO USE THIS CLASS:
    
    # first you need to make inheritance of this class
    class My_GA(GA_Base):
        # implement every SHOULD BE OVERIDEN METHODS
    
    # make instance and do things
    ga = My_GA()
    ga.elitism_rate = 0.4
    ga.process()
    ga.show()
    
    '''


    def __init__(self):
        '''
        Constructor        
        '''
        self._individuals = [] # array of dictionary
        self._fitness = [] # array of dictionary
        self._generations = []
        self._max_epoch = 100
        self._population_size = 100
        self._representations = ['default']
        self._benchmarks = ['default']
        self._elitism_rate = 0.2
        self._mutation_rate = 0.3
        self._crossover_rate = 0.3
        self._new_rate = 0.2
    
    def take_individual(self,generation_index):
        return {}
    
    def do_elitism(self):
        return []
    
    def do_mutation(self,individual):
        return {}
    
    def do_crossover(self,individual_1, individual_2):
        return {}
    
    def do_process_individual(self,individual):
        return {}
    
    def calculate_fitness(self,individual):
        return {'default':0}
    
    def register_individual(self,individual):
        individual = self._do_process_individual(individual)
        try:          
            return self._individuals.index(individual)            
        except:
            fitness = self._do_calculate_fitness(individual)
            self._individuals.append(individual)
            self._fitness.append(fitness)
            return self._individuals.index(individual)
    
    def process(self):
        total_rate = self._elitism_rate + self._mutation_rate + self._crossover_rate + self._new_rate
        elitism_count = int(self._population_size * self._elitism_rate/total_rate)
        mutation_count = elitism_count + int(self._population_size * self._mutation_rate/total_rate)
        crossover_count = mutation_count + int(self._population_size * self._crossover_rate/total_rate)
        # for every generation
        gen = 0
        while gen<self._max_epoch:
            # for every population of generation
            pop = 0
            while pop<self._population_size:
                if pop<elitism_count:
                    pass
                elif pop<mutation_count:
                    pass
                elif pop<crossover_count:
                    pass
                else:
                    pass
                pop+=1
            gen+=1
    
    def show(self):
        benchmarks = self._benchmarks
        generation_index = np.arange(len(self._generations)) 
        # max_fitnesses and min_fitnesses of each generation for every benchmark
        max_fitnesses = {}
        min_fitnesses = {}          
        for benchmark in benchmarks:
            max_fitnesses[benchmark] = []
            min_fitnesses[benchmark] = [] 
        # variation of each generation
        variations = []   
        for i in xrange(generation_index):
            max_fitness = {}
            min_fitness = {}
            unique_index = []
            for j in xrange(self._generations[i]):
                if not j in unique_index:
                    unique_index.append(j)
                for benchmark in benchmarks:
                    if j == 0:
                        max_fitness[benchmark] = self._fitness[j][benchmark]
                        min_fitness[benchmark] = self._fitness[j][benchmark]
                    else:
                        if self._fitness[j][benchmark] > max_fitness[benchmark]:
                            max_fitness[benchmark] = self._fitness[j][benchmark]
                        if self._fitness[j][benchmark] < min_fitness[benchmark]:
                            min_fitness[benchmark] = self._fitness[j][benchmark]
            max_fitnesses[benchmark].append(max_fitness[benchmark])
            min_fitnesses[benchmark].append(min_fitness[benchmark])
            variations.append(len(unique_index))
                    
        fig = plt.figure()
        # maximum
        sp_1 = fig.add_subplot(2,2,1)
        sp_1.set_title('Maximum Fitness of Generations')
        sp_1.set_y_label('Fitness Value')
        sp_1.set_x_label('Generation')
        for benchmark in benchmarks:
            sp_1.plot(generation_index, max_fitnesses[benchmark])
        # minimum
        sp_2 = fig.add_subplot(2,2,2)
        sp_2.set_title('Minimum Fitness of Generations')
        sp_2.set_y_label('Fitness Value')
        sp_2.set_x_label('Generation')
        for benchmark in benchmarks:
            sp_2.plot(generation_index, min_fitnesses[benchmark])
        # variation
        sp_3 = fig.add_subplot(2,2,3)
        sp_3.set_title('Minimum Fitness of Generations')
        sp_3.set_y_label('Fitness Value')
        sp_3.set_x_label('Generation')
        sp_3.plot(generation_index, variations)
        fig.show()
    
    @property
    def individuals(self):
        return self._individuals
    
    @property
    def fitness(self):
        return self._fitness
    
    @property
    def generations(self):
        return self._generations
    
    @property
    def max_epoch(self):
        return self._max_epoch
    @max_epoch.setter
    def max_epoch(self,value):
        self._max_epoch = int(value)
    
    @property
    def population_size(self):
        return self._population_size
    @population_size.setter
    def population_size(self,value):
        self._population_size = int(value)
    
    @property
    def elitism_rate(self):
        return self._elitism_rate
    @elitism_rate.setter
    def elitism_rate(self,value):
        self._elitism_rate = float(value)
    
    @property
    def mutation_rate(self):
        return self._mutation_rate
    @mutation_rate.setter
    def mutation_rate(self,value):
        self._mutation_rate = float(value)
    
    @property
    def crossover_rate(self):
        return self._crossover_rate
    @crossover_rate.setter
    def crossover_rate(self,value):
        self._crossover_rate = float(value)
    
    @property
    def new_rate(self):
        return self._new_rate
    @new_rate.setter
    def new_rate(self,value):
        self._new_rate = float(value)
    
    @property
    def representations(self):
        return self._representations
    @representations.setter
    def representations(self,value):
        if type(value) == list:
            self._representations = value
        else:
            self._representations.append(value)
            
    @property
    def benchmarks(self):
        return self._benchmarks
    @benchmarks.setter
    def benchmarks(self,value):
        if type(value) == list:
            self._benchmarks = value
        else:
            self._benchmarks.append(value)
        