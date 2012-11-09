'''
Created on Nov 7, 2012

@author: gofrendi
'''

from Base import bin_digit_needed, bin_to_dec
from GA import Genetics_Algorithm

class Grammatical_Evolution(Genetics_Algorithm):
    
    def __init__(self):
        super(Grammatical_Evolution, self).__init__()
        self.representations = ['default', 'phenotype']
        self._variables = []
        self._grammar = {}
        self._start_node = 'expr'
        
    def execute(self, expr, record):
        result = 0
        error = False
        # get result and error state
        try:
            sandbox={}
            # initialize features
            for i in xrange(len(self._variables)):
                feature = self._variables[i]       
                exec(feature+' = '+str(record[i])) in sandbox 
            # execute expr, and get the result         
            exec('__result = '+expr) in sandbox                      
            result = sandbox['__result']
        except:
            error = True    
        return result, error
    
    def _transform(self, gene):
        depth = 10
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
                        digit_needed = bin_digit_needed(possibility)
                        # if end of gene, then start over from the beginning
                        if(gene_index+digit_needed)>len(gene):
                            gene_index = 0
                        # get part of gene that will be used
                        used_gene = gene[gene_index:gene_index+digit_needed]
                        if(used_gene == ''):
                            print gene, gene_index, digit_needed, len(gene)  
                        
                        gene_index = gene_index + digit_needed                          
                                               
                        rule_index = bin_to_dec(used_gene) % possibility
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
