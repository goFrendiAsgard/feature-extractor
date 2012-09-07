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
__version__     = "0.0.2"
__maintainer__  = "Go Frendi Gunawan"
__status__      = "Development"

# ==================================0000====================================== #


import random
random.seed(1567)

# This is the class declaration
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
    def _to_decimal(binary_number):
        # manual way is verbose and not elegant, but in this case speed is
        # very important.
        binary_dictionary = {
            '0'   : 0,   '1'   : 1,   '00'  : 0,   '01'  : 1,
            '10'  : 2,   '11'  : 3,   '000' : 0,   '001' : 1,
            '010' : 2,   '011' : 3,   '100' : 4,   '101' : 5,
            '110' : 6,   '111' : 7
        }        
        try:
            number = binary_dictionary[binary_number]
            return number
        except:
            # since power operation is expensive, it should only be used in 
            # very special case
            multiplier = 0
            number = 0
            for el in binary_number[0:-1]:
                number += int(el) * (2**multiplier)
                multiplier += 1
            return number
    
    # return how many binary digit needed to represent a decimal number
    def _get_binary_digit_count(self, number):
        # manual way is verbose and not elegant, but in this case speed is
        # very important. 
        power_dictionary = {
            '2'   : 1,   '4'   : 1,   '8'   : 3,   '16'  : 4,
            '32'  : 5,   '64'  : 6,   '128' : 7,   '256' : 8
        }     
        for key in power_dictionary:
            if number<int(key):
                return power_dictionary[key]
            
        # since power operation is expensive, it should only be used in very
        # special case
        i = 8
        while number < 2**i :
            i += 1
        return i
        
    
    # return phenotype of the gene
    def _transform(self, gene):
        depth = 10
        gene_index = 0
        newExpr = self.start_node
        # for each level
        for level in xrange(depth):
            i=0
            # parse every character in the expr
            while i<len(newExpr):
                find = False
                for key in self.grammar:
                    # if there is a keyword in the grammar, replace it
                    # with rule in production
                    if (expr[i:i+len(key)] == key):
                        find = True
                        # count how many transformation possibility exists
                        possibility = len(self.grammar[key])
                        # calculate how many binary digit needed to represent
                        # the possibilities
                        digit_needed = self.get_binary_digit(possibility)
                        used_gene = gene[gene_index:digit_needed]
                        # cyclic gene
                        if(gene_index+used_gene)>len(gene):
                            gene_index = 0
                        else: 
                            gene_index += self.get_binary_digit(possibility)
                        rule_index = self._to_decimal(used_gene)
                        newExpr += self.grammar[key][rule_index]
                        i+= len(key)-1
                if not find:
                    newExpr += expr[i:i+1]
                i += 1
        return newExpr
    
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
    
    # new individu
    def _new_individu(self):
        pass
    
    # new generation
    def _new_generation(self):
        pass
    
    def process(self):
        # the original features has a VIP chance to be in competition
        for i in range(len(self.original_features)):
            phenotype = self.original_features[i]
            try:
                test = self.phenotype_fitness[phenotype]
            except:
                self.phenotype_fitness[phenotype] = self._calculate_fitness(phenotype)
            
                
        # other processes
        
        # show the results
        print('Genotype-Phenotype Dictionary :')
        print(self.genotype_dictionary)
        print('Fitness of All Phenotypes :')
        print(self.phenotype_fitness)
        
        
 
# =============================== TEST CASE ================================== #
if __name__ == '__main__':
 
    # declare parameters
    features = ['<x>','<y>']
    start_node = '<expr>'
    grammar = {
        start_node : [
            '<value> <operator> <value>', 
            '<value>', 
            '<function>(<expr>)'],
        '<value>' : ['<feature>', '<number>'],
        '<feature>' : features,
        '<number>' : ['<digit>.<digit>', '<digit>'],
        '<digit>' : ['<digit><digit>', '0', '1', '2', '3', '4', '5', '6', '7',
            '8', '9'],
        '<opreator>' : ['+', '-', '*', '/'],
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
