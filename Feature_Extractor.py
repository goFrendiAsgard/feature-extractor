#!/usr/bin/env python

# ==================================0000====================================== #

""" 
    Feature_Extractor.py: 
    A program to proof my hypothesis about feature extraction 
    by using grammatical evolution with multiple fitness value
"""

__author__      = "Go Frendi Gunawan"
__copyright__   = "Copyright 2012, Planet Earth, Solar System"
__credits__     = ["Go Frendi Gunawan"]
__license__     = "GPL"
__version__     = "0.0.3"
__maintainer__  = "Go Frendi Gunawan"
__status__      = "Development"

# ==================================0000====================================== #


import random
random.seed(15)

# ============================================================================ # 
# =========================== FEATURE EXTRACTOR ============================== #
# ============================================================================ #

class Feature_Extractor(object):
    
    # ------------------------------ constructor ----------------------------- #
    def __init__(self):
        # attributes 
        self.genotype_dictionary = {}
        self.phenotype_fitness = {}
        self.original_features = []
        self.grammar = {}
        self.start_node = ''
        self.classes = []
        self.data = []
        self.gene_length = 50
        self.population_size = 200
    
    # -------------------------------- setters ------------------------------- #
    
    # set features
    def set_features(self, features):
        self.original_features = features
    
    # set grammar
    def set_grammar(self, grammar):
        self.grammar = grammar
    
    # set start node
    def set_start_node(self, start_node):
        self.start_node = start_node
    
    # set classes
    def set_classes(self, classes):
        self.classes = classes
    
    # set data
    def set_data(self, data):
        self.data = data 
    
    # -------------------------------- methods ------------------------------- #
    
    # binary to decimal conversion
    def _to_decimal(self, binary_number):
        # manual way is verbose and not elegant, but in this case speed is very important.
        binary_dictionary = {
            '0'   : 0,   '1'   : 1,   '00'  : 0,   '01'  : 1,
            '10'  : 2,   '11'  : 3,   '000' : 0,   '001' : 1,
            '010' : 2,   '011' : 3,   '100' : 4,   '101' : 5,
            '110' : 6,   '111' : 7,   '0000': 0,   '0001': 1,
            '0010': 2,   '0011': 3,   '0100': 4,   '0101': 5,
            '0110': 6,   '0111': 7,   '1000': 8,   '1001': 9,
            '1010': 10,  '1011':11,   '1100': 12,  '1101': 13,
            '1110': 14,  '1111':15
        }        
        try:
            number = binary_dictionary[binary_number]
            return number
        except:
            # since power operation is expensive, it should only be used in very special case
            multiplier = 0
            number = 0
            for el in binary_number[0:-1]:
                number += int(el) * (2**multiplier)
                multiplier += 1
            return number
    
    # return how many binary digit needed to represent a decimal number
    def _get_binary_digit_count(self, number):
        # manual way is verbose and not elegant, but in this case speed is very important. 
        power_array = [
            [2   , 1 ], [4   , 2 ], [8   , 3 ], [16  , 4 ], [32  , 5 ],
            [64  , 6 ], [128 , 7 ], [256 , 8 ], [512 , 9 ], [1024, 10],
        ]
        for i in range(len(power_array)):
            if number<=int(power_array[i][0]):
                return power_array[i][1]
            
        # since power operation is expensive, it should only be used in very special case
        i = 8
        while number < 2**i :
            i += 1
        return i
        
    
    # return phenotype of the gene
    def _transform(self, gene):
        depth = 10
        gene_index = 0
        expr = self.start_node
        # for each level
        level = 0
        while level < depth:
            i=0
            new_expr = ''
            # parse every character in the expr
            while i<len(expr):                
                found = False
                for key in self.grammar:
                    # if there is a keyword in the grammar, replace it with rule in production
                    if (expr[i:i+len(key)] == key):
                        found = True
                        # count how many transformation possibility exists
                        possibility = len(self.grammar[key])
                        # how many binary digit needed to represent the possibilities
                        digit_needed = self._get_binary_digit_count(possibility)
                        used_gene = gene[gene_index:gene_index+digit_needed]
                        # if end of gene, then start over from the beginning
                        if(gene_index+len(used_gene))>len(gene):
                            gene_index = 0
                        else: 
                            gene_index += digit_needed                        
                        rule_index = self._to_decimal(used_gene) % possibility
                        new_expr += self.grammar[key][rule_index]
                        i+= len(key)-1
                if not found:
                    new_expr += expr[i:i+1]
                i += 1
            expr = new_expr
            level = level+1
        return expr
    
    # execute expression
    def _execute(self, expr, record):
        result = 0
        error = False
        # get result and error state
        try:
            sandbox={}
            # initialize features
            for i in xrange(len(self.original_features)):
                feature = self.original_features[i]       
                exec(feature+' = '+str(record[i])) in sandbox 
            # execute expr, and get the result         
            exec('__result = '+expr) in sandbox                      
            result = sandbox['__result']
        except:
            error = True    
        return result, error
    
    # calculate how the data projected in new feature
    def _calculate_projection(self, phenotype):
        projection = {}
        feature_count = len(self.original_features)
        for label in self.classes:
            projection[label] = {
                'values' : [],
                'minimum' : 0,
                'maximum' : 0,
                'error' : False
            }
        for record in self.data:
            result = self._execute(phenotype, record)
            label = record[feature_count]
            if result[1]:
                projection[label]['error'] = True
            elif projection[label]['error'] == False:
                value = result[0]
                if len(projection[label]['values'])==0:
                    projection[label]['minimum'] = value
                    projection[label]['maximum'] = value
                else:
                    if value<projection[label]['minimum']:
                        projection[label]['minimum'] = value
                    if value>projection[label]['maximum']:
                        projection[label]['maximum'] = value
                projection[label]['values'].append(value) 
        return projection    
    
    # calculate the fitness value of a phenotype
    def _calculate_fitness(self, phenotype):
        projection = self._calculate_projection(phenotype)
        fitness = {}
        for label in self.classes:
            if projection[label]['error']:
                fitness[label] = 0
                continue
            else:
                between_count = 0
                overlap_count = 0
                for other_label in self.classes:
                    if other_label == label:
                        continue
                    if projection[other_label]['error']:
                        continue                    
                    for other_value in projection[other_label]['values']:
                        if other_value >= projection[label]['minimum'] and other_value<= projection[label]['maximum']:
                            between_count += 1
                        for value in projection[label]['values']:
                            if value == other_value:
                                overlap_count += 1
                fitness[label] = 1/( overlap_count*0.1 + between_count*0.01 + 0.001)
        return fitness
    
    # register genotype into genotype_dictionary and fenotype_fitness
    def _register_genotype(self, gene):
    
        # declare phenotype
        phenotype = ''        
        
        # if the gene is not exists is genotype_dictionary, then calculate
        # phenotype, and register it in genotype_dictionary.
        # else, just take phenotype value from genotype_dictionary
        if gene in self.genotype_dictionary:
            phenotype = self.genotype_dictionary[gene]
        else:
            phenotype = self._transform(gene)
            self.genotype_dictionary[gene] = phenotype          
        
        # register the phenotype fitness
        if not (phenotype in self.phenotype_fitness):
                self.phenotype_fitness[phenotype] = self._calculate_fitness(phenotype)
    
    # return new individu's genotype (binary string)
    def _new_individu(self):
        gene = ''
        i = 0
        while i < self.gene_length:
            number = random.randint(0,10)
            if number<5:
                gene += '0'
            else:
                gene += '1'
            i = i+1
        return gene
    
    # return new generation (array of binary string)
    def _new_generation(self):
        generation = []
        i = 0
        while i < self.population_size:
            individu = self._new_individu()
            generation.append(individu)
            i = i+1
        return generation
    
    def process(self):
        # the original features has a VIP chance to be in competition
        for i in range(len(self.original_features)):
            phenotype = self.original_features[i]
            if not (phenotype in self.phenotype_fitness):
                self.phenotype_fitness[phenotype] = self._calculate_fitness(phenotype) 
                
        # new generation
        generation = self._new_generation()
        for i in range(len(generation)):
            self._register_genotype(generation[i])
        
        # show phenotypes
        print('')
        print('# ============ Phenotype List (%d) : ============ #' %(len(self.phenotype_fitness)) )
        for fitness in self.phenotype_fitness:
            print('%s : %s' %(fitness, self.phenotype_fitness[fitness]))
        print('# ============= End of Phenotype List ============ #')
        print('')
        
        # search the best for it's kind
        best_values = {}
        best_phenotype = {}
        for label in self.classes:
            for fitness in self.phenotype_fitness:
                if (not (label in best_values)) or (best_values[label]<self.phenotype_fitness[fitness][label]):
                    best_values[label] = self.phenotype_fitness[fitness][label]
                    best_phenotype[label] = fitness
            print('%s : %s ,fitness: %s' %(label, best_phenotype[label], best_values[label]) )
        print('')
        
        
# ============================================================================ # 
# =============================== TEST CASE ================================== #
# ============================================================================ #
if __name__ == '__main__':
 
    # declare parameters
    features = ['sepal_length','sepal_width','petal_length','petal_width']
    start_node = '<expr>'
    grammar = {
        start_node : [
            '<expr> <operator> <expr>',
            '<number> <operator> <feature>',
            '<feature> <operator> <number>',
            '<function>(<expr>)',
            '<feature>'],
        '<feature>' : features,
        '<number>' : ['<digit>.<digit>', '<digit>'],
        '<digit>' : [
            '<digit><digit>', 
            '0', '1', '2', '3', '4', 
            '5', '6', '7', '8', '9'],
        '<operator>' : ['+', '-', '*', '/', '**'],
        '<function>' : ['sin', 'cos', 'tan', 'abs']
    }
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    data = [
            [5.1,3.5,1.4,0.2,'Iris-setosa'],
            [4.9,3.0,1.4,0.2,'Iris-setosa'],
            [4.7,3.2,1.3,0.2,'Iris-setosa'],
            [4.6,3.1,1.5,0.2,'Iris-setosa'],
            [5.0,3.6,1.4,0.2,'Iris-setosa'],
            [5.4,3.9,1.7,0.4,'Iris-setosa'],
            [4.6,3.4,1.4,0.3,'Iris-setosa'],
            [5.0,3.4,1.5,0.2,'Iris-setosa'],
            [4.4,2.9,1.4,0.2,'Iris-setosa'],
            [4.9,3.1,1.5,0.1,'Iris-setosa'],
            [5.4,3.7,1.5,0.2,'Iris-setosa'],
            [4.8,3.4,1.6,0.2,'Iris-setosa'],
            [4.8,3.0,1.4,0.1,'Iris-setosa'],
            [4.3,3.0,1.1,0.1,'Iris-setosa'],
            [5.8,4.0,1.2,0.2,'Iris-setosa'],
            [5.7,4.4,1.5,0.4,'Iris-setosa'],
            [5.4,3.9,1.3,0.4,'Iris-setosa'],
            [5.1,3.5,1.4,0.3,'Iris-setosa'],
            [5.7,3.8,1.7,0.3,'Iris-setosa'],
            [5.1,3.8,1.5,0.3,'Iris-setosa'],
            [5.4,3.4,1.7,0.2,'Iris-setosa'],
            [5.1,3.7,1.5,0.4,'Iris-setosa'],
            [4.6,3.6,1.0,0.2,'Iris-setosa'],
            [5.1,3.3,1.7,0.5,'Iris-setosa'],
            [4.8,3.4,1.9,0.2,'Iris-setosa'],
            [5.0,3.0,1.6,0.2,'Iris-setosa'],
            [5.0,3.4,1.6,0.4,'Iris-setosa'],
            [5.2,3.5,1.5,0.2,'Iris-setosa'],
            [5.2,3.4,1.4,0.2,'Iris-setosa'],
            [4.7,3.2,1.6,0.2,'Iris-setosa'],
            [4.8,3.1,1.6,0.2,'Iris-setosa'],
            [5.4,3.4,1.5,0.4,'Iris-setosa'],
            [5.2,4.1,1.5,0.1,'Iris-setosa'],
            [5.5,4.2,1.4,0.2,'Iris-setosa'],
            [4.9,3.1,1.5,0.1,'Iris-setosa'],
            [5.0,3.2,1.2,0.2,'Iris-setosa'],
            [5.5,3.5,1.3,0.2,'Iris-setosa'],
            [4.9,3.1,1.5,0.1,'Iris-setosa'],
            [4.4,3.0,1.3,0.2,'Iris-setosa'],
            [5.1,3.4,1.5,0.2,'Iris-setosa'],
            [5.0,3.5,1.3,0.3,'Iris-setosa'],
            [4.5,2.3,1.3,0.3,'Iris-setosa'],
            [4.4,3.2,1.3,0.2,'Iris-setosa'],
            [5.0,3.5,1.6,0.6,'Iris-setosa'],
            [5.1,3.8,1.9,0.4,'Iris-setosa'],
            [4.8,3.0,1.4,0.3,'Iris-setosa'],
            [5.1,3.8,1.6,0.2,'Iris-setosa'],
            [4.6,3.2,1.4,0.2,'Iris-setosa'],
            [5.3,3.7,1.5,0.2,'Iris-setosa'],
            [5.0,3.3,1.4,0.2,'Iris-setosa'],
            [7.0,3.2,4.7,1.4,'Iris-versicolor'],
            [6.4,3.2,4.5,1.5,'Iris-versicolor'],
            [6.9,3.1,4.9,1.5,'Iris-versicolor'],
            [5.5,2.3,4.0,1.3,'Iris-versicolor'],
            [6.5,2.8,4.6,1.5,'Iris-versicolor'],
            [5.7,2.8,4.5,1.3,'Iris-versicolor'],
            [6.3,3.3,4.7,1.6,'Iris-versicolor'],
            [4.9,2.4,3.3,1.0,'Iris-versicolor'],
            [6.6,2.9,4.6,1.3,'Iris-versicolor'],
            [5.2,2.7,3.9,1.4,'Iris-versicolor'],
            [5.0,2.0,3.5,1.0,'Iris-versicolor'],
            [5.9,3.0,4.2,1.5,'Iris-versicolor'],
            [6.0,2.2,4.0,1.0,'Iris-versicolor'],
            [6.1,2.9,4.7,1.4,'Iris-versicolor'],
            [5.6,2.9,3.6,1.3,'Iris-versicolor'],
            [6.7,3.1,4.4,1.4,'Iris-versicolor'],
            [5.6,3.0,4.5,1.5,'Iris-versicolor'],
            [5.8,2.7,4.1,1.0,'Iris-versicolor'],
            [6.2,2.2,4.5,1.5,'Iris-versicolor'],
            [5.6,2.5,3.9,1.1,'Iris-versicolor'],
            [5.9,3.2,4.8,1.8,'Iris-versicolor'],
            [6.1,2.8,4.0,1.3,'Iris-versicolor'],
            [6.3,2.5,4.9,1.5,'Iris-versicolor'],
            [6.1,2.8,4.7,1.2,'Iris-versicolor'],
            [6.4,2.9,4.3,1.3,'Iris-versicolor'],
            [6.6,3.0,4.4,1.4,'Iris-versicolor'],
            [6.8,2.8,4.8,1.4,'Iris-versicolor'],
            [6.7,3.0,5.0,1.7,'Iris-versicolor'],
            [6.0,2.9,4.5,1.5,'Iris-versicolor'],
            [5.7,2.6,3.5,1.0,'Iris-versicolor'],
            [5.5,2.4,3.8,1.1,'Iris-versicolor'],
            [5.5,2.4,3.7,1.0,'Iris-versicolor'],
            [5.8,2.7,3.9,1.2,'Iris-versicolor'],
            [6.0,2.7,5.1,1.6,'Iris-versicolor'],
            [5.4,3.0,4.5,1.5,'Iris-versicolor'],
            [6.0,3.4,4.5,1.6,'Iris-versicolor'],
            [6.7,3.1,4.7,1.5,'Iris-versicolor'],
            [6.3,2.3,4.4,1.3,'Iris-versicolor'],
            [5.6,3.0,4.1,1.3,'Iris-versicolor'],
            [5.5,2.5,4.0,1.3,'Iris-versicolor'],
            [5.5,2.6,4.4,1.2,'Iris-versicolor'],
            [6.1,3.0,4.6,1.4,'Iris-versicolor'],
            [5.8,2.6,4.0,1.2,'Iris-versicolor'],
            [5.0,2.3,3.3,1.0,'Iris-versicolor'],
            [5.6,2.7,4.2,1.3,'Iris-versicolor'],
            [5.7,3.0,4.2,1.2,'Iris-versicolor'],
            [5.7,2.9,4.2,1.3,'Iris-versicolor'],
            [6.2,2.9,4.3,1.3,'Iris-versicolor'],
            [5.1,2.5,3.0,1.1,'Iris-versicolor'],
            [5.7,2.8,4.1,1.3,'Iris-versicolor'],
            [6.3,3.3,6.0,2.5,'Iris-virginica'],
            [5.8,2.7,5.1,1.9,'Iris-virginica'],
            [7.1,3.0,5.9,2.1,'Iris-virginica'],
            [6.3,2.9,5.6,1.8,'Iris-virginica'],
            [6.5,3.0,5.8,2.2,'Iris-virginica'],
            [7.6,3.0,6.6,2.1,'Iris-virginica'],
            [4.9,2.5,4.5,1.7,'Iris-virginica'],
            [7.3,2.9,6.3,1.8,'Iris-virginica'],
            [6.7,2.5,5.8,1.8,'Iris-virginica'],
            [7.2,3.6,6.1,2.5,'Iris-virginica'],
            [6.5,3.2,5.1,2.0,'Iris-virginica'],
            [6.4,2.7,5.3,1.9,'Iris-virginica'],
            [6.8,3.0,5.5,2.1,'Iris-virginica'],
            [5.7,2.5,5.0,2.0,'Iris-virginica'],
            [5.8,2.8,5.1,2.4,'Iris-virginica'],
            [6.4,3.2,5.3,2.3,'Iris-virginica'],
            [6.5,3.0,5.5,1.8,'Iris-virginica'],
            [7.7,3.8,6.7,2.2,'Iris-virginica'],
            [7.7,2.6,6.9,2.3,'Iris-virginica'],
            [6.0,2.2,5.0,1.5,'Iris-virginica'],
            [6.9,3.2,5.7,2.3,'Iris-virginica'],
            [5.6,2.8,4.9,2.0,'Iris-virginica'],
            [7.7,2.8,6.7,2.0,'Iris-virginica'],
            [6.3,2.7,4.9,1.8,'Iris-virginica'],
            [6.7,3.3,5.7,2.1,'Iris-virginica'],
            [7.2,3.2,6.0,1.8,'Iris-virginica'],
            [6.2,2.8,4.8,1.8,'Iris-virginica'],
            [6.1,3.0,4.9,1.8,'Iris-virginica'],
            [6.4,2.8,5.6,2.1,'Iris-virginica'],
            [7.2,3.0,5.8,1.6,'Iris-virginica'],
            [7.4,2.8,6.1,1.9,'Iris-virginica'],
            [7.9,3.8,6.4,2.0,'Iris-virginica'],
            [6.4,2.8,5.6,2.2,'Iris-virginica'],
            [6.3,2.8,5.1,1.5,'Iris-virginica'],
            [6.1,2.6,5.6,1.4,'Iris-virginica'],
            [7.7,3.0,6.1,2.3,'Iris-virginica'],
            [6.3,3.4,5.6,2.4,'Iris-virginica'],
            [6.4,3.1,5.5,1.8,'Iris-virginica'],
            [6.0,3.0,4.8,1.8,'Iris-virginica'],
            [6.9,3.1,5.4,2.1,'Iris-virginica'],
            [6.7,3.1,5.6,2.4,'Iris-virginica'],
            [6.9,3.1,5.1,2.3,'Iris-virginica'],
            [5.8,2.7,5.1,1.9,'Iris-virginica'],
            [6.8,3.2,5.9,2.3,'Iris-virginica'],
            [6.7,3.3,5.7,2.5,'Iris-virginica'],
            [6.7,3.0,5.2,2.3,'Iris-virginica'],
            [6.3,2.5,5.0,1.9,'Iris-virginica'],
            [6.5,3.0,5.2,2.0,'Iris-virginica'],
            [6.2,3.4,5.4,2.3,'Iris-virginica'],
            [5.9,3.0,5.1,1.8,'Iris-virginica']
        ]
    
    # create Feature_Extractor, and set it's parameter
    fe = Feature_Extractor()
    fe.set_features(features)
    fe.set_start_node(start_node)
    fe.set_grammar(grammar)
    fe.set_classes(classes)
    fe.set_data(data)
    
    # show the result
    fe.process()
