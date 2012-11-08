'''
Created on Nov 7, 2012

@author: gofrendi
'''

from Base import GA_Base

class Grammatical_Evolution(GA_Base):
    
    def __init__(self):
        super(Grammatical_Evolution, self).__init__()
        self.representations = ['default', 'phenotype']
        self._variables = []
        self._grammar = {}
        self._start_node = ''
        
    def _execute(self, expr, record):
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
        #TODO: adjust this
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
    
    def do_calculate_fitness(self, individual):
        return GA_Base.do_calculate_fitness(self, individual)
    
    def do_crossover(self, individual_1, individual_2):
        return GA_Base.do_crossover(self, individual_1, individual_2)
    
    def do_mutation(self, individual):
        return GA_Base.do_mutation(self, individual)
    
    def do_generate_new_individual(self):
        return GA_Base.do_generate_new_individual(self)
    
    def do_process_individual(self, individual):
        return GA_Base.do_process_individual(self, individual)
