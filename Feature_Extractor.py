# ========== Feature Extractor By Using Grammatical Evolution With =========== # 
# ====================== Multi Fitness Calculation ============================#
#                                                                              #
#  Author : Go Frendi Gunawan                                                  #
#  Description : Proofing my final project hypothesis about feature extraction #
#                                                                              #
# ============================================================================ #

import random
random.seed(1567)

# This is the class declaration
class Feature_Extractor(object):
    
    # constructor
    def __init__(self):
        self.genotype_dictionary = []
        self.fenotype_fitness = []
        self.original_features = []
        self.grammar = {}
        self.start_node = ''
        self.classes = []
        self.data = []
    
    # setters ==================================================================
    
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
    
    # methods ==================================================================
    
    # mutation
    def _mutation(self, gene):
        pass
    
    # crossover
    def _crossover(self, gene_1, gene_2):
        pass
    
    # new individu
    def _new_individu(self):
        pass
    
    def _new_generation(self):
        pass
 
# Test case ====================================================================
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
        '<number>' : ['<digit>.<digit>', '<digit>']
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
