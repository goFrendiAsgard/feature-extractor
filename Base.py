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

randomizer = random.Random(10)
    

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
        self._fitness_measurement = 'MAX'
        self._individual_benchmark_rank = {} # rank of individual in current generation
        self._individual_total_fitness = {} # total fitness of individuals
    
    def _sort(self,benchmark,lst=None):
        '''
        Quick sorting self._individual_benchmark_rank
        '''
        if lst is None:
            lst = self._individual_benchmark_rank[benchmark]
        if len(lst)==0:
            return lst
        else:
            pivot = lst[0]
            lesser = self._sort(benchmark, [x for x in lst[1:] if x['fitness'] < pivot['fitness']])
            greater = self._sort(benchmark, [x for x in lst[1:] if x['fitness'] >= pivot['fitness']])
            if self._fitness_measurement == 'MAX':
                return greater + [pivot] + lesser                
            else:
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
            indexes = []
            if generation_index<(self._max_epoch-1):
                indexes = self._generations[generation_index]
            else:
                indexes = xrange(len(self._individuals))
            for individual_index in indexes:
                individual_benchmark = {}
                individual_benchmark['index'] = individual_index
                individual_benchmark['fitness'] = self._fitness[individual_index][benchmark]
                self._individual_benchmark_rank[benchmark].append(individual_benchmark)
                self._individual_total_fitness[benchmark] += float(self._fitness[individual_index][benchmark])
            # quick sort
            self._individual_benchmark_rank[benchmark] = self._sort(benchmark)
            
    
    def _get_random_individual_indexes(self,benchmark='default'):
        # take random individual (roulette wheel scenario for a benchmark)
        num = randomizer.random() * self._individual_total_fitness[benchmark]
        acc = 0
        for i in xrange(len(self._individual_benchmark_rank[benchmark])):
            acc += self._individual_benchmark_rank[benchmark][i]['fitness']
            if acc>= num:
                return self._individual_benchmark_rank[benchmark][i]['index']
    
    def _get_elite_individual_indexes(self,count,benchmark='default'):
        individuals = []
        for i in xrange(count):
            individuals.append(self._individual_benchmark_rank[benchmark][i]['index'])
        return individuals
    
    def _add_to_generation(self,individual_index, generation_index=0):
        while generation_index > (len(self._generations)-1):
            self._generations.append([])
        self._generations[generation_index].append(individual_index)
        
    
    def _register_individual(self,individual, generation_index=0):
        # completing individual representation
        individual = self._process_individual(individual)
        if individual in self._individuals:
            # add to generation      
            individual_index = self._individuals.index(individual)
            self._add_to_generation(individual_index, generation_index)            
        else:
            # calculation fitness
            fitness = self._calculate_fitness(individual)
            # register individual and fitness as 'already calculated'
            # so we don't need to calculate it again whenever meet such an individual
            self._individuals.append(individual)
            self._fitness.append(fitness)
            # add to generation
            individual_index = self._individuals.index(individual)
            self._add_to_generation(individual_index, generation_index)
    
    def process(self):
        benchmark_count = len(self._benchmarks)
        total_rate = self._elitism_rate + self._mutation_rate + self._crossover_rate + self._new_rate
        elitism_count = int(self._population_size * self._elitism_rate/total_rate)
        mutation_count = int(self._population_size * self._mutation_rate/total_rate)
        crossover_count = int(self._population_size * self._crossover_rate/total_rate)
        elite_individual_per_benchmark = elitism_count/benchmark_count or 1
        mutation_individual_per_benchmark = mutation_count/benchmark_count or 1
        crossover_individual_per_benchmark = crossover_count/benchmark_count or 1
        
        # adjust population size
        minimum_population_size = benchmark_count * (elite_individual_per_benchmark+mutation_individual_per_benchmark+crossover_individual_per_benchmark)        
        if self._population_size<minimum_population_size:
            self._population_size = minimum_population_size
        
        print('Elite individual : %d' %(elite_individual_per_benchmark * benchmark_count))
        print('Mutation Individual : %d' %(mutation_individual_per_benchmark * benchmark_count))
        print('Crossover Individual : %d' %(crossover_individual_per_benchmark * benchmark_count))
        
        # for every generation
        gen = 0
        while gen<self._max_epoch:            
            if gen==0:
                self._generations.append([])
            else:
                # for every benchmark
                for benchmark in self._benchmarks:
                    # _get_elite_individual_indexes
                    individual_indexes = self._get_elite_individual_indexes(elite_individual_per_benchmark, benchmark)
                    for individual_index in individual_indexes:
                        self._add_to_generation(individual_index, gen)
                    # mutation
                    for i in xrange(mutation_individual_per_benchmark):
                        individual_index = self._get_random_individual_indexes(benchmark)
                        individual = self._individuals[individual_index]
                        new_individual = self._mutation(individual)
                        self._register_individual(new_individual, gen)
                    # crossover
                    for i in xrange(int(crossover_individual_per_benchmark/2) or 1):
                        individual_1_index = self._get_random_individual_indexes(benchmark)
                        individual_2_index = self._get_random_individual_indexes(benchmark)
                        individual_1 = self._individuals[individual_1_index]
                        individual_2 = self._individuals[individual_2_index]
                        new_individual_1, new_individual_2 = self._crossover(individual_1, individual_2)
                        self._register_individual(new_individual_1, gen)
                        self._register_individual(new_individual_2, gen)
            i = len(self._generations[gen])            
            # fill out the current generation with new individuals
            while i<self._population_size:
                individual = self._generate_new_individual()
                self._register_individual(individual, gen)
                i+=1
            # process the population
            self._process_population(gen)
            
            # print the output
            print('Generation %d' % (gen+1))
            
            gen+=1
    
    def best_individuals(self,count=1,benchmark='default',representation='default'):
        if count==1:
            index = self._individual_benchmark_rank[benchmark][0]['index']
            return self._individuals[index][representation]
        else:
            representations = []
            for i in xrange(count):
                index = self._individual_benchmark_rank[benchmark][i]['index']
                representations.append(self._individuals[index][representation])
            return representations
    
    def show(self):
        benchmarks = self._benchmarks
        generation_indexes = np.arange(len(self._generations)) 
        # max_fitnesses and min_fitnesses of each generation for every benchmark
        max_fitnesses = {}
        min_fitnesses = {}          
        for benchmark in benchmarks:
            max_fitnesses[benchmark] = []
            min_fitnesses[benchmark] = [] 
        # variation of each generation
        variations = []
        most_minimum = 0
        most_maximum = 0
        for i in xrange(len(generation_indexes)):
            max_fitness = {}
            min_fitness = {}
            unique_index = []
            for j in xrange(len(self._generations[i])):
                index = self._generations[i][j]
                if not index in unique_index:
                    unique_index.append(index)
                for benchmark in benchmarks:
                    if j == 0:
                        max_fitness[benchmark] = self._fitness[index][benchmark]
                        min_fitness[benchmark] = self._fitness[index][benchmark]
                        if i == 0:
                            most_minimum = min_fitness[benchmark]
                            most_maximum = max_fitness[benchmark]
                    else:
                        if self._fitness[index][benchmark] > max_fitness[benchmark]:
                            max_fitness[benchmark] = self._fitness[index][benchmark]
                        if self._fitness[index][benchmark] < min_fitness[benchmark]:
                            min_fitness[benchmark] = self._fitness[index][benchmark]
                    if min_fitness[benchmark] < most_minimum:
                        most_minimum = min_fitness[benchmark]
                    if max_fitness[benchmark] > most_maximum:
                        most_maximum = max_fitness[benchmark]                      
            for benchmark in benchmarks:
                max_fitnesses[benchmark].append(max_fitness[benchmark])
                min_fitnesses[benchmark].append(min_fitness[benchmark])
            variations.append(len(unique_index))
            
        fig_y_range = most_maximum - most_minimum
        min_y = most_minimum - (fig_y_range * 0.5)
        max_y = most_maximum + (fig_y_range * 0.5)
                    
        fig = plt.figure()
        # maximum
        sp_1 = fig.add_subplot(2,2,1)
        sp_1.set_title('Maximum Fitness')
        sp_1.set_ylabel('Fitness Value')
        sp_1.set_xlabel('Generation')
        sp_1.set_ylim(min_y,max_y)      
        for benchmark in benchmarks:
            sp_1.plot(generation_indexes, max_fitnesses[benchmark], label=benchmark)
        sp_1.legend(shadow=True, loc=0)
        # minimum
        sp_2 = fig.add_subplot(2,2,2)
        sp_2.set_title('Minimum Fitness')
        sp_2.set_ylabel('Fitness Value')
        sp_2.set_xlabel('Generation')
        sp_2.set_ylim(min_y,max_y) 
        for benchmark in benchmarks:
            sp_2.plot(generation_indexes, min_fitnesses[benchmark], label=benchmark)
        sp_2.legend(shadow=True, loc=0)
        # variation
        sp_3 = fig.add_subplot(2,2,3)
        sp_3.set_title('Variations')
        sp_3.set_ylabel('Individual Variation')
        sp_3.set_xlabel('Generation')
        sp_3.plot(generation_indexes, variations)
        #adjust subplot
        plt.subplots_adjust(hspace = 0.5, wspace = 1)
        plt.show()
    
    def _generate_new_individual(self):
        new_individual = self.do_generate_new_individual()
        # check type
        if not type(new_individual) is dict:
            new_individual={}
        return new_individual
    
    def _mutation(self,individual):
        individual = dict(individual)
        new_individual = self.do_mutation(individual)
        # check type
        if not type(new_individual) is dict:
            new_individual={}
        return new_individual
    
    def _crossover(self,individual_1,individual_2):
        individual_1 = dict(individual_1)
        individual_2 = dict(individual_2)
        new_individual_1, new_individual_2 = self.do_crossover(individual_1,individual_2)
        # check type
        if not type(new_individual_1) is dict:
            new_individual_1={}
        if not type(new_individual_2) is dict:
            new_individual_2={}
        return new_individual_1, new_individual_2
    
    def _process_individual(self,individual):
        individual = dict(individual)
        individual = self.do_process_individual(individual)
        # check type
        if not type(individual) is dict:
            individual={}
        # check key
        for representation in self._representations:
            if not representation in individual:
                individual[representation] = 0.0
        for representation in individual:
            if not representation in self._representations:
                self._representations.append(representation)
        return individual
    
    def _calculate_fitness(self,individual):
        individual = dict(individual)
        fitness = self.do_calculate_fitness(individual) 
        # check type
        if not type(fitness) is dict:
            fitness={}
        # check key
        for benchmark in self._benchmarks:
            if not benchmark in fitness:
                fitness[benchmark] = 0.0
        for benchmark in fitness:
            if not benchmark in self._benchmarks:
                self._benchmarks.append(benchmark)
        return fitness
    
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
    def fitness_measurement(self):
        return self._fitness_measurement
    @fitness_measurement.setter
    def fitness_measurement(self, value):
        if type(value) is str:
            if value.upper() == 'MAX':
                self._fitness_measurement = 'MAX'
            else:
                self._fitness_measurement = 'MIN'
        else:
            self._fitness_measurement = 'MAX'
    
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