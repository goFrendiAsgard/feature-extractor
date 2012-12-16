'''
Created on Nov 10, 2012

@author: gofrendi
'''

import utils, gc
import matplotlib.pyplot as plt
import numpy as np

class Tree(object):
    def __init__(self):
        self._data = ''
        self._children = []
    
    def get_random_node(self):
        if self.children_count>0:
            child_index = utils.randomizer.randrange(0,self.children_count)
            rnd = utils.randomizer.randrange(0,2)
            if rnd==0:
                return self.get_child(child_index)
            else:
                return self.get_child(child_index).get_random_node()
        else:
            return self
    
    def get_child(self,index):
        return self.children[index]
    
    def replace_child(self,index,value):
        self._children[index] = value
    
    def add_child(self,value=None):
        if type(value) is str:
            child = Tree()
            child.data = value
            self._children.append(child)
        elif type(value) is Tree:
            child = value
            self._children.append(child)
        else:
            child = Tree()
            self._children.append(child)
    
    def remove_child(self,index):
        del(self.children[index])
    
    def as_program(self):
        program = ''
        if self.children_count==0:
            program = self._data
        else:
            program = self._data + '('
            for i in xrange(len(self._children)):
                child = self._children[i]
                program += child.as_program()
                if i<len(self._children)-1:
                    program += ','                
            program += ')'
        return program
    
    def generate(self,nodes=[['x','y'],[],['plus','minus','multiply','divide']],max_level = 10):
        children_count = 0
        if max_level > 0:
            children_count = utils.randomizer.randrange(0,len(nodes))
        # anticipate if there is an empty array among nodes
        while len(nodes[children_count])==0:
            if children_count<len(nodes)-1:
                children_count += 1
            else:
                children_count = 0
        # generate node_index
        node_index = utils.randomizer.randrange(0,len(nodes[children_count]))
        self.data = nodes[children_count][node_index]
        self.children = []
        if children_count>0:
            for i in xrange(children_count):
                self.add_child()
                child = self.get_child(i)
                child.generate(nodes, max_level-1)
    
    def __repr__(self, *args, **kwargs):
        return self.as_program()
    
    @property
    def children_count(self):
        return len(self.children)
    
    @property
    def children(self):
        return self._children
    @children.setter
    def children(self, value):
        self._children = value
    
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self,value):
        self._data = value

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
        self._assumpted_individuals = []
        self._generations = []
        self._max_epoch = 50
        self._population_size = 100
        self._representations = ['default']
        self._benchmarks = ['default']
        self._elitism_rate = 0.2
        self._mutation_rate = 0.4
        self._crossover_rate = 0.3
        self._new_rate = 0.1
        self._fitness_measurement = 'MAX'
        self._stopping_value = None
        self._individual_benchmark_rank = {} # rank of individual in current generation
        self._individual_total_fitness = {} # total fitness of individuals
        self._label = ''
        self._convergance_rate = 0.3
    
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
            equal = [x for x in lst[1:] if x['fitness'] == pivot['fitness']]
            greater = self._sort(benchmark, [x for x in lst[1:] if x['fitness'] > pivot['fitness']])
            if self._fitness_measurement == 'MAX':
                return greater + [pivot] + equal + lesser                
            else:
                return lesser + equal + [pivot] + greater        
    
    def _process_population(self,generation_index, end=False):
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
            if not end:
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
        '''
        take random individual (roulette wheel scenario for a benchmark)
        '''
        num = utils.randomizer.random() * self._individual_total_fitness[benchmark]
        acc = 0
        for i in xrange(len(self._individual_benchmark_rank[benchmark])):
            acc += self._individual_benchmark_rank[benchmark][i]['fitness']
            if acc>= num:
                return self._individual_benchmark_rank[benchmark][i]['index']
        return self._individual_benchmark_rank[benchmark][len(self._individual_benchmark_rank[benchmark])-1]['index']
    
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
        utils.write("Processing %s" % (self.label))
                
        benchmark_count = len(self.benchmarks)
        total_rate = self._elitism_rate + self._mutation_rate + self._crossover_rate + self._new_rate
        elitism_count = int(self._population_size * self._elitism_rate/total_rate)
        mutation_count = int(self._population_size * self._mutation_rate/total_rate)
        crossover_count = int(self._population_size * self._crossover_rate/total_rate)
        elite_individual_per_benchmark = elitism_count/benchmark_count or 1
        mutation_individual_per_benchmark = mutation_count/benchmark_count or 1
        crossover_individual_per_benchmark = crossover_count/benchmark_count or 1
        
        # stopping value
        if type(self.stopping_value) is dict:
            for benchmark in self.benchmarks:                
                if not benchmark in self.stopping_value:
                    self.stopping_value[benchmark] = None
            for benchmark in self.stopping_value:
                if not benchmark in self.benchmarks:
                    del(self.stopping_value[benchmark])
        elif not self.stopping_value is None:
            stopping_value = float(self.stopping_value)
            self.stopping_value = {}
            for benchmark in self.benchmarks:
                self.stopping_value[benchmark] = stopping_value
        
        # adjust population size
        minimum_population_size = benchmark_count * (elite_individual_per_benchmark+mutation_individual_per_benchmark+crossover_individual_per_benchmark)        
        if self._population_size<minimum_population_size:
            self._population_size = minimum_population_size     
        
        
        self._generations.append([])
        
        # add individuals based on assumptions
        if self.assumpted_individuals is None:
            self.assumpted_individuals=[]
        if len(self.assumpted_individuals)>0:
            for i in xrange(min(len(self.assumpted_individuals), self.population_size)):
                new_individual =  self._process_individual(self.assumpted_individuals[i])
                self._register_individual(new_individual, 0)        
        
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
            
            ended = False
            # check stopping condition
            if not self.stopping_value is None:
                can_stop = True
                self._process_population(gen, ended)
                for benchmark in self.benchmarks:
                    if self.stopping_value[benchmark] is None:
                        continue
                    else:
                        index = self._individual_benchmark_rank[benchmark][0]['index']
                        fitness = self.fitness[index][benchmark]
                        if self.fitness_measurement == 'MAX':
                            if fitness < self.stopping_value[benchmark]:
                                can_stop = False
                                break
                        else:
                            if fitness > self.stopping_value[benchmark]:
                                can_stop = False
                                break
                ended = can_stop
            
            # check convergance
            unique_individuals = []
            for individual in self.generations[gen]:
                if not individual in unique_individuals:
                    unique_individuals.append(individual)
            unique_individual_count = len(unique_individuals)
            if unique_individual_count<= self.convergance_rate * self.population_size:
                ended = True
                        
            ended = ended or gen==(self._max_epoch-1)
            
            # process the population
            self._process_population(gen, ended)
            
            # print the output
            utils.write("Processing generation %d of %s" % (gen+1, self.label))
            
            # break if ended
            if ended:
                break
            
            gen+=1
        utils.write("Done processing %s" % (self.label))
        print('')
    
    def best_individuals(self,count=1,benchmark='default',representation='default'):
        if count==1:
            index = self._individual_benchmark_rank[benchmark][0]['index']
            return self.individuals[index][representation]
        else:
            representations = []
            for i in xrange(count):
                index = self._individual_benchmark_rank[benchmark][i]['index']
                representations.append(self.individuals[index][representation])
            return representations
    
    def best_fitnesses(self,count=1,benchmark='default'):
        if count==1:
            index = self._individual_benchmark_rank[benchmark][0]['index']
            return self.fitness[index][benchmark]
        else:
            fitnesses = []
            for i in xrange(count):
                index = self._individual_benchmark_rank[benchmark][i]['index']
                fitnesses.append(self.fitnesses[index][benchmark])
            return fitnesses
    
    def show(self, silent=False, file_name='figure.png'):
        benchmarks = self._benchmarks
        generation_indexes = np.arange(len(self._generations))
        
        # create figure
        fig = plt.figure(figsize=(20.0, 12.0))
        benchmark_count = len(self.benchmarks)
        subplot_index = 1
        
        # minimum and maximum x for each benchmark equavalent to the count of generations
        x_range = max(len(generation_indexes)-1, 1.0)
        min_x = 1 - 0.01 * x_range
        max_x = len(generation_indexes) + 0.01 * x_range
        # for each benchmark, draw its max and min graphics
        for benchmark in benchmarks:
            # get maximum and minimum
            max_fitnesses = []
            min_fitnesses = []
            max_max_fitness = 0
            min_max_fitness = 0
            max_min_fitness = 0
            min_min_fitness = 0
            for i in xrange(len(generation_indexes)):
                max_fitness = 0
                min_fitness = 0
                for j in xrange(len(self._generations[i])):
                    index = self._generations[i][j]
                    fitness = self.fitness[index][benchmark]
                    if j==0:
                        max_fitness = fitness
                        min_fitness = fitness
                    else:
                        if fitness > max_fitness:
                            max_fitness = fitness
                        if fitness < min_fitness:
                            min_fitness = fitness
                # add max & min fitness to max_fitnesses and min_fitnesses
                max_fitnesses.append(max_fitness)
                min_fitnesses.append(min_fitness)
                if i==0:
                    max_max_fitness = max_fitness
                    min_max_fitness = max_fitness
                    max_min_fitness = min_fitness
                    min_min_fitness = min_fitness
                else:
                    if max_fitness > max_max_fitness:
                        max_max_fitness = max_fitness
                    if max_fitness < min_max_fitness:
                        min_max_fitness = max_fitness
                    if min_fitness > max_min_fitness:
                        max_min_fitness = min_fitness
                    if min_fitness < min_min_fitness:
                        min_min_fitness = min_fitness
                        
            # maximum benchmark subplot
            sp = fig.add_subplot(benchmark_count+1, 2, subplot_index)
            subplot_index += 1
            sp.set_title('Maximum Fitness for '+str(benchmark))
            sp.set_ylabel('Fitness Value')
            sp.set_xlabel('Generation Index')
            if len(generation_indexes) == 1:
                sp.plot(generation_indexes+1, max_fitnesses, 'bo')
            else:
                sp.plot(generation_indexes+1, max_fitnesses)
            y_range = max(max_max_fitness - min_max_fitness, 1.0)
            min_y = min_max_fitness - 0.05 * y_range
            max_y = max_max_fitness + 0.05 * y_range
            sp.set_ylim(min_y, max_y)
            sp.set_xlim(min_x, max_x)
            
            # minimum benchmark subplot
            sp = fig.add_subplot(benchmark_count+1, 2, subplot_index)
            subplot_index += 1
            sp.set_title('Minimum Fitness for '+str(benchmark))
            sp.set_ylabel('Fitness Value')
            sp.set_xlabel('Generation Index')
            if len(generation_indexes) == 1:
                sp.plot(generation_indexes+1, min_fitnesses,'bo')
            else:
                sp.plot(generation_indexes+1, min_fitnesses)
            y_range = max(max_min_fitness - min_min_fitness, 1.0)
            min_y = min_min_fitness - 0.05 * y_range
            max_y = max_min_fitness + 0.05 * y_range
            sp.set_ylim(min_y, max_y)
            sp.set_xlim(min_x, max_x)
        
        variations = []
        max_variation = 0
        min_variation = 0
        for i in xrange(len(generation_indexes)):
            unique_individuals = []
            for j in xrange(len(self._generations[i])):
                index = self._generations[i][j]
                if not index in unique_individuals:
                    unique_individuals.append(index)
            variation = len(unique_individuals)
            if i==0:
                max_variation = variation
                min_variation = variation
            else:
                if max_variation < variation:
                    max_variation = variation
                if min_variation > variation:
                    min_variation = variation
            variations.append(variation)
        # individual variation
        sp = fig.add_subplot(benchmark_count+1, 2, subplot_index)
        subplot_index += 1
        sp.set_title('Individual Variation')
        sp.set_ylabel('Unique individual')
        sp.set_xlabel('Generation Index')
        if len(generation_indexes) == 1:
            sp.plot(generation_indexes+1, variations,'bo')
        else:
            sp.plot(generation_indexes+1, variations)
        y_range = max(max_variation - min_variation, 1.0)
        min_y = min_variation - 0.05 * y_range
        max_y = max_variation + 0.05 * y_range
        sp.set_ylim(min_y, max_y)
        sp.set_xlim(min_x, max_x)
        
        #adjust subplot
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        plt.suptitle('%s, Fitness Measurement : %s' %(self.label, self.fitness_measurement))
        
        if silent:
            plt.savefig(file_name, dpi=100)
            print('save plot to '+file_name)
        else:
            plt.show()
            print('show figure')
        
        fig.clf()
        plt.close()
        gc.collect()
    
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
        if type(value) is list:
            self._representations = value
        else:
            self._representations.append(value)
            
    @property
    def benchmarks(self):
        return self._benchmarks
    @benchmarks.setter
    def benchmarks(self,value):
        if type(value) is list:
            self._benchmarks = value
        else:
            self._benchmarks.append(value)
    
    @property
    def stopping_value(self):
        return self._stopping_value
    @stopping_value.setter
    def stopping_value(self,value):
        self._stopping_value = value
    
    @property
    def label(self):
        return self._label
    @label.setter
    def label(self, value):
        self._label = value
    
    @property
    def assumpted_individuals(self):
        return self._assumpted_individuals
    @assumpted_individuals.setter
    def assumpted_individuals(self,value):
        self._assumpted_individuals = value
    
    @property
    def convergance_rate(self):
        return self._convergance_rate
    @convergance_rate.setter
    def convergance_rate(self, value):
        self._convergance_rate = value

class Genetics_Algorithm(GA_Base):
    '''
    Genetics Algorithm class
    '''
    
    def __init__(self):
        super(Genetics_Algorithm, self).__init__()
        self._individual_length = 10
        self.label = 'Genetics Algorithm'
        
    def do_process_individual(self, individual):
        return individual
    
    def do_generate_new_individual(self):
        individual = {}
        individual['default'] = ''
        while len(individual['default'])<self._individual_length:
            rnd = utils.randomizer.randrange(0,2)
            individual['default'] = individual['default'] + str(rnd)
        return individual
    
    def do_crossover(self, individual_1, individual_2):
        rnd = utils.randomizer.randrange(min(len(individual_1['default']),len(individual_2['default'])))
        gene_1 = individual_1['default'][:rnd]
        gene_2 = individual_1['default'][rnd:]
        gene_3 = individual_2['default'][:rnd]
        gene_4 = individual_2['default'][rnd:]
        individual_1['default'] = gene_1 + gene_4
        individual_2['default'] = gene_2 + gene_3
        return individual_1, individual_2
    
    def do_mutation(self, individual):
        rnd = utils.randomizer.randrange(self._individual_length)
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

class Grammatical_Evolution(Genetics_Algorithm):
    
    def __init__(self):
        super(Grammatical_Evolution, self).__init__()
        self.label = 'Grammatical Evolution'
        self.representations = ['default', 'phenotype']
        self._variables = ['x','y']
        self._grammar = {
            'expr':['var','expr op expr'],
            'var' :['x','y'],
            'op'  :['+','-','*','/']
        }
        self._start_node = 'expr'
    
    def _transform(self, gene):
        depth = 20
        gene_index = 0
        expr = self._start_node
        # for each level
        level = 0
        while level < depth:
            i=0
            new_expr = ''
            # parse every character in the expr
            while i<len(expr):                
                found = False
                for key in self._grammar:
                    # if there is a keyword in the grammar, replace it with rule in production
                    if (expr[i:i+len(key)] == key):
                        found = True
                        # count how many transformation possibility exists
                        possibility = len(self._grammar[key])
                        # how many binary digit needed to represent the possibilities
                        digit_needed = utils.bin_digit_needed(possibility)
                        # if end of gene, then start over from the beginning
                        if(gene_index+digit_needed)>len(gene):
                            gene_index = 0
                        # get part of gene that will be used
                        used_gene = gene[gene_index:gene_index+digit_needed]
                        
                        gene_index = gene_index + digit_needed                          
                                               
                        rule_index = utils.bin_to_dec(used_gene) % possibility
                        new_expr += self._grammar[key][rule_index]
                        i+= len(key)-1
                if not found:
                    new_expr += expr[i:i+1]
                i += 1
            expr = new_expr
            level = level+1
        return expr
    
    def do_process_individual(self, individual):
        genotype = individual['default']
        phenotype = self._transform(genotype)
        individual['phenotype'] = phenotype
        return individual
    
    @property
    def grammar(self):
        return self._grammar
    @grammar.setter
    def grammar(self,value):
        self._grammar = value
    
    @property
    def variables(self):
        return self._variables
    @variables.setter
    def variables(self,value):
        self._variables = value
    
    @property
    def start_node(self):
        return self._start_node
    @start_node.setter
    def start_node(self,value):
        self._start_node = value

class Genetics_Programming(GA_Base):
    def __init__(self):
        super(Genetics_Programming,self).__init__()
        self.label = 'Genetics Programming'
        self.representation = ['default','phenotype']
        self._nodes = [['x','y'],[],['plus','minus','multiply','divide']]
    
    def do_process_individual(self, individual):
        individual['phenotype'] = individual['default'].as_program()
        return individual
    
    def do_generate_new_individual(self):
        tree = Tree()
        tree.generate(self._nodes)
        individual = {'default':tree}
        return individual
    
    def do_crossover(self, individual_1, individual_2):
        node_1 = individual_1['default'].get_random_node()
        node_2 = individual_2['default'].get_random_node()
        node_1, node_2 = node_2, node_1
        return individual_1, individual_2
    
    def do_mutation(self, individual):
        node = individual['default'].get_random_node()
        children_count = node.children_count
        rnd = utils.randomizer.randrange(0,len(self._nodes[children_count]))
        node.data = self._nodes[children_count][rnd]
        return individual
    
    @property
    def nodes(self):
        return self._nodes
    @nodes.setter
    def nodes(self,value):
        self._nodes = value