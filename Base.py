'''
Created on Oct 23, 2012

@author: gofrendi
'''

import random
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
        self._individual_benchmark_rank = {} # rank of individual in current generation
        self._individual_total_fitness = {} # total fitness of individuals
    
    def _sort(self,benchmark):
        '''
        Quick sorting self._individual_benchmark_rank
        '''
        lst = self._individual_benchmark_rank[benchmark]
        pivot = lst[0]
        lesser = self._sort([x for x in lst[1:] if x['fitness'] < pivot['fitness']])
        greater = self._sort([x for x in lst[1:] if x['fitness'] >= pivot['fitness']])
        return lesser + [pivot] + greater
    
    def _process_population(self,generation_index):
        '''
        Process the population
        making self._individual_benchmark_rank and self._individual_total_fitness
        '''
        self._individual_benchmark_rank = {}
        self._individual_total_fitness = {}
        for benchmark in self._benchmarks:
            self._individual_benchmark_rank[benchmark] = []
            self._individual_total_fitness[benchmark] = 0.0
            for individual_index in self._generations[generation_index]:
                individual_benchmark = {}
                individual_benchmark['index'] = individual_index
                individual_benchmark['fitness'] = self._fitness[individual_index][benchmark]
                self._individual_benchmark_rank[benchmark].append(individual_benchmark)
                self._individual_total_fitness += self._fitness[individual_index][benchmark]
            # quick sort
            self._individual_benchmark_rank[benchmark] = self._sort(self,benchmark)
            
    
    def _get_individual_indexes(self,benchmark='default'):
        # take random individual (roulette wheel scenario for a benchmark)
        num = random.random() * self._individual_total_fitness[benchmark]
        acc = 0
        for i in xrange(self._individual_benchmark_rank):
            acc += self._individual_benchmark_rank[i]['fitness']
            if acc>= num:
                return self._individual_benchmark_rank['index']
    
    def _get_elite_individual_indexes(self,count,benchmark='default'):
        individuals = []
        for i in xrange(count):
            individuals.append(self._individual_benchmark_rank[benchmark][i]['index'])
        return individuals
    
    def _add_to_generation(self,individual_index, generation_index=0):
        if generation_index < (len(self._generations)-1):
            self._generations[generation_index] = []
        self._generations[generation_index].append(individual_index)
        
    
    def _register_individual(self,individual, generation_index=0):
        # completing individual representation
        individual = self._do_process_individual(individual)
        try:    
            # add to generation      
            individual_index = self._individuals.index(individual)
            self._add_to_generation(individual_index, generation_index)            
        except:
            # calculation fitness
            fitness = self._do_calculate_fitness(individual)
            # register individual and fitness as 'already calculated'
            # so we don't need to calculate it again whenever meet such an individual
            self._individuals.append(individual)
            self._fitness.append(fitness)
            # add to generation
            individual_index = self._individuals.index(individual)
            self._add_to_generation(individual_index, generation_index)
    
    def process(self):
        total_rate = self._elitism_rate + self._mutation_rate + self._crossover_rate + self._new_rate
        elitism_count = int(self._population_size * self._elitism_rate/total_rate)
        mutation_count = elitism_count + int(self._population_size * self._mutation_rate/total_rate)
        crossover_count = mutation_count + int(self._population_size * self._crossover_rate/total_rate)
        elite_individual_per_benchmark = elitism_count/len(self._benchmarks) or 1
        mutation_individual_per_benchmark = mutation_count/len(self._benchmarks) or 1
        crossover_individual_per_benchmark = crossover_count/len(self._benchmarks) or 1
        # for every generation
        gen = 0
        while gen<self._max_epoch:
            if gen>0:
                # for every benchmark
                for benchmark in self._benchmarks:
                    # _get_elite_individual_indexes
                    individual_indexes = self._get_elite_individual_indexes(elite_individual_per_benchmark, benchmark)
                    for individual_index in individual_indexes:
                        self._add_to_generation(individual_index, gen)
                    # mutation
                    for i in xrange(mutation_individual_per_benchmark):
                        individual = self._get_individual_indexes(benchmark)
                        new_individual = self.do_mutation(individual)
                        self._register_individual(new_individual, gen)
                    # crossover
                    for i in xrange(int(crossover_individual_per_benchmark/2) or 1):
                        individual_1 = self._get_individual_indexes(benchmark)
                        individual_2 = self._get_individual_indexes(benchmark)
                        new_individual_1, new_individual_2 = self.do_crossover(individual_1, individual_2)
                        self._register_individual(new_individual_1, gen)
                        self._register_individual(new_individual_2, gen)
            i = len(self._generations[gen])
            # fill out the current generation with new individuals
            while i<self._population_size:
                individual = self.do_generate_new_individual()
                self._register_individual(individual, gen)
                i+=1
            # process the population
            self._process_population(gen)
                
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
    
    def do_generate_new_individual(self):
        '''
        This method should return a new individual
        '''
        raise NotImplementedError
        return {}
    
    def do_mutation(self,individual):
        '''
        This method should return an individual after mutation
        '''
        raise NotImplementedError
        return {}
    
    def do_crossover(self,individual_1, individual_2):
        '''
        This method should return 2 individuals after crossover process
        '''
        raise NotImplementedError
        return []
    
    def do_process_individual(self,individual):
        '''
        This method should return complete individual
        '''
        raise NotImplementedError
        return {}
    
    def do_calculate_fitness(self,individual):
        '''
        This method should return fitness of individual
        '''
        return {'default':0}
    
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