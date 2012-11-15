'''
Created on Nov 13, 2012

@author: gofrendi
'''
import classes
import sys
import utils
from sklearn import svm

class GP_SVM(classes.Genetics_Algorithm):
    pass

class GE_SVM(classes.Grammatical_Evolution):
    pass

class GP_Global_Fitness(classes.Genetics_Programming):
    pass

class GE_Global_Fitness(classes.Grammatical_Evolution):
    pass

class GP_Multi_Fitness(classes.Genetics_Programming):
    pass

class GE_Multi_Fitness(classes.Grammatical_Evolution):
    
    def __init__(self):
        self.records = []
    
    def do_calculate_fitness(self, individual):
        utils.execute(expr, record, variables)
        return classes.Grammatical_Evolution.do_calculate_fitness(self, individual)

class Feature_Extractor(object):
    def __init__(self):
        self.records = []
        self.variables = []
        
    def process(self):
        # using svm
        gp_svm = GP_SVM()
        gp_svm.label = 'Genetics Programming with SVM'
        
        ge_svm = GE_SVM()
        ge_svm.label = 'Grammatical Evolution with SVM'
        
        # using one fitness measurement
        gp_global_fitness = GP_Global_Fitness()
        gp_global_fitness.label = 'Genetics Programming Global Fitness'
        
        ge_global_fitness = GE_Global_Fitness()
        ge_global_fitness.label = 'Grammatical Evolution Global Fitness'
        
        # using multi fitness measurement
        gp_multi_fitness = GP_Multi_Fitness()
        gp_multi_fitness.label = 'Genetics Programming Multi Fitness'
        
        ge_multi_fitness = GE_Multi_Fitness()
        ge_multi_fitness.label = 'Grammatical Evolution Multi Fitness'
        
        
        # process & show
        extractors = [gp_svm, ge_svm, gp_global_fitness, ge_global_fitness, gp_multi_fitness, ge_multi_fitness]
        for extractor in extractors:
            extractor.process()
            extractor.show()

if __name__ == '__main__':
    sys.argv
    fe = Feature_Extractor()
    fe.records = [[0,0,'salah'],[0,1,'benar'],[1,0,'benar'],[1,1,'benar']]
    fe.variables=['x','y']
    fe.process()
