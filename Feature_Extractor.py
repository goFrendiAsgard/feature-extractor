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
        self.population_size = 1000
    
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
        for level in xrange(depth):
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
    
    # calculate the fitness value of a phenotype
    def _calculate_fitness(self, phenotype):
        for classes in self.classes:
            for record in self.data:
                pass
    
    # register genotype into genotype_dictionary and fenotype_fitness
    def _register_genotype(self, gene):
    
        # declare phenotype
        phenotype = ''        
        
        # if the gene is not exists is genotype_dictionary, then calculate
        # phenotype, and register it in genotype_dictionary.
        # else, just take phenotype value from genotype_dictionary
        try:
            phenotype = self.genotype_dictionary[gene]
        except:
            phenotype = self._transform(gene)
            self.genotype_dictionary[gene] = phenotype            
        
        # register the phenotype fitness
        try:
            test = self.phenotype_fitness[phenotype]
        except:
            self.phenotype_fitness[phenotype] = self._calculate_fitness(phenotype)
    
    # return new individu's genotype (binary string)
    def _new_individu(self):
        gene = ''
        for i in range(self.gene_length):
            number = random.randint(0,10)
            if number<5:
                gene += '0'
            else:
                gene += '1'
        return gene
    
    # return new generation (array of binary string)
    def _new_generation(self):
        generation = []
        for i in range(self.population_size):
            individu = self._new_individu()
            generation.append(individu)
        return generation
    
    def process(self):
        # the original features has a VIP chance to be in competition
        for i in range(len(self.original_features)):
            phenotype = self.original_features[i]
            try:
                test = self.phenotype_fitness[phenotype]
            except:
                self.phenotype_fitness[phenotype] = self._calculate_fitness(phenotype)
            
                
        # new generation
        generation = self._new_generation()
        for i in range(len(generation)):
            self._register_genotype(generation[i])
        
        # show the results
        print('')
        print('# ============ Phenotype List (%d) : ============ #' %(len(self.phenotype_fitness)) )
        for fitness in self.phenotype_fitness:
            print('%s : %s' %(fitness, self.phenotype_fitness[fitness]))
        print('# ============= End of Phenotype List ============ #')
        print('')
        
        
# ============================================================================ # 
# =============================== TEST CASE ================================== #
# ============================================================================ #
if __name__ == '__main__':
 
    # declare parameters
    features = ['x','y']
    start_node = '<expr>'
    grammar = {
        start_node : [
            '<value> <operator> <value>',
            '<function>(<expr>)'],
        '<value>' : ['<feature>', '<number>'],
        '<feature>' : features,
        '<number>' : ['<digit>.<digit>', '<digit>'],
        '<digit>' : [
            '<digit><digit>', 
            '0', '1', '2', '3', '4', 
            '5', '6', '7', '8', '9'],
        '<operator>' : ['+', '-', '*', '/', '**'],
        '<function>' : ['sin', 'cos', 'tan', 'abs']
    }
    classes = ['A', 'B', 'C']
    data = [
        [0, 0, 'A'],
        [1, 1, 'A'],
        [2, 2, 'B'],
        [3, 3, 'B'],
        [4, 4, 'C'],
        [5, 5, 'C']
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
