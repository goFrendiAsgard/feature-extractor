'''
Created on Nov 13, 2012

@author: gofrendi
'''
import classes, utils, numpy
from sklearn import svm, datasets
import matplotlib.pyplot as plt

def count_unmatch(data_1, data_2):
    if len(data_1) == len(data_2) and len(data_1)>0:
        count = 0
        for i in xrange(len(data_1)):
            if data_1[i] <> data_2[i]:
                count+=1
        return count
    else:
        raise TypeError

def calculate_mse(data_1, data_2):
    if (len(data_1) == len(data_2)) and len(data_1)>0:
        se = 0.0
        for i in xrange(len(data_1)):
            se += (data_1[i] - data_2[i]) ** 2
        mse = se/len(data_1)
        return mse
    else:
        raise TypeError

def get_comparison(data_1, data_2):
    return 'Unmatch = %d, MSE = %f' %(count_unmatch(data_1,data_2), calculate_mse(data_1,data_2))

def select_feature(data, mask):
    new_data = []
    if type(data[0]) is list:
        for i in xrange(len(data)):
            new_record = []
            for j in xrange(len(data[i])):
                if mask[j] == '1':
                    new_record.append(data[i][j])
            new_data.append(new_record)
    else:
        for i in xrange(len(data)):
            if mask[i] == '1':
                new_data.append(data[i])
    return new_data

def get_feature_count(data):
    return len(data[0])

def show_svm(training_data, training_target, test_data, test_target, mask=[], variables=[], label='', svc=None):
    if svc is None:
        svc = svm.SVC(kernel='linear').fit(training_data, training_target)    
    if len(mask)>0:
        training_data = select_feature(training_data, mask)
        test_data = select_feature(test_data, mask)
        variables = select_feature(variables, mask)
        
    svc_training_prediction = svc.predict(training_data)
    svc_test_prediction = svc.predict(test_data)
    label = ' '+label+' '
    limitter_count = int((90-len(label))/2)
    print('='*limitter_count + label + '='*limitter_count)
    print('TRAINING      : '+get_comparison(training_target,svc_training_prediction))
    print('TEST          : '+get_comparison(test_target,svc_test_prediction))
    print('FEATURE COUNT : '+str(get_feature_count(training_data)))
    print('USED FEATURE  : '+", ".join(variables))
    

class GA_SVM(classes.Genetics_Algorithm): 
    def __init__(self):
        super(GA_SVM,self).__init__()
        self.fitness_measurement = 'MIN'
        self.representations=['default','svc', 'unmatch_count']
        self.benchmarks=['unmatch_count', 'mse']
        self.individual_length=20
        self.training_data = []
        self.training_target = []
    
    def process(self):
        if not len(self.training_data) == 0 and not len(self.training_data[0]) == 0:
            self.individual_length = len(self.training_data[0])+6
        else:
            self.individual_length = 20
        classes.Genetics_Algorithm.process(self)
    
    def do_process_individual(self, individual):
        gene = individual['default']
        # if the dataset is empty, then everything is impossible, don't try to do anything
        if not len(self.training_data) == 0 and not len(self.training_data[0]) == 0:
            feature_count = len(self.training_data[0]) # how many feature available in the data
            
            # feature selection by using first digits of the gene
            new_training_data = select_feature(self.training_data, gene)
            
            # perform svm (actually we still able to give some improvements, like choosing kernel etc)
            kernel_gene = gene[feature_count:feature_count+2]
            degree_gene = gene[feature_count+2:feature_count+4]
            gamma_gene = gene[feature_count+4:feature_count+6]
            kernel_option = {'00':'linear', '01':'linear', '10':'rbf', '11':'poly'}
            degree_value = utils.bin_to_dec(degree_gene)+1
            gamma_value = float(utils.bin_to_dec(gamma_gene)+1)/10.0
            svc = svm.SVC(kernel=kernel_option[kernel_gene], C=1.0, degree=degree_value, gamma=gamma_value).fit(new_training_data, self.training_target)
            prediction = svc.predict(new_training_data)
            individual['svc'] = svc
            individual['unmatch_count'] = count_unmatch(self.training_target, prediction)
            individual['mse'] = calculate_mse(self.training_target, prediction)
        return individual
           
    def do_calculate_fitness(self, individual):                    
        return {'unmatch_count': individual['unmatch_count'], 'mse':individual['mse']}

class GE_Multi_Fitness(classes.Grammatical_Evolution):
    def __init__(self):
        super(GE_Multi_Fitness, self).__init__()
        self.fitness_measurement = 'MIN'
        self.variables = []
        self.grammar = {
            '<expr>' : ['<var>','<expr> <op> <expr>','<func>(<expr>)'],
            '<var>'  : self.variables,
            '<op>'   : ['+','-','*','/'],
            '<func>' : ['sqr','sqrt']
        }
        self.start_node = '<expr>'
        self.training_data = []
        self.training_target = []
        self.classes = []
    
    def process(self):
        # adjust grammar & benchmark
        self.grammar['<var>'] = self.variables
        self.benchmarks = self.classes
        # cheating, add some assumpted individuals, ensure that the original ones are involved in competition
        gene_var_prefix = '00'
        variable_count = len(self.variables)
        digit_needed = utils.bin_digit_needed(variable_count)
        for i in xrange(variable_count):
            gene_var = utils.dec_to_bin(i)
            while len(gene_var)<digit_needed:
                gene_var = '0'+gene_var
            gene = gene_var_prefix + gene_var
            while len(gene)<self.individual_length:
                gene += '0'
            self.assumpted_individuals.append({'default':gene})
        # do process as nothing happens before :)
        classes.Grammatical_Evolution.process(self)
    
    def do_calculate_fitness(self, individual):
        training_data = self.training_data
        training_target = self.training_target
        phenotype = individual['phenotype']
        variables = self.variables
        projection = {}
        fitness = {}
        exec_error = {}
        min_projection = {}
        max_projection = {}
        for benchmark in self.benchmarks:
            projection[benchmark] = []             
            fitness[benchmark] = 0
            exec_error[benchmark] = False
            min_projection[benchmark] = 0
            max_projection[benchmark] = 0
        # calculate projection
        for i in xrange(len(training_data)):
            record = training_data[i]
            for benchmark in self.benchmarks:
                if benchmark == training_target[i]:
                    result, error = utils.execute(phenotype, record, variables)
                    if error:
                        exec_error[benchmark] = True
                        continue
                    else:
                        result = float(result)
                        if len(projection[benchmark]) == 0:
                            min_projection[benchmark] = result
                            max_projection[benchmark] = result
                        else:
                            if result<min_projection[benchmark]:
                                min_projection[benchmark] = result
                            if result>max_projection[benchmark]:
                                max_projection[benchmark] = result
                        projection[benchmark].append(result)                        
                    break
        # calculate fitnesses
        for current_benchmark in self.benchmarks:
            
            if exec_error[current_benchmark]:
                fitness[current_benchmark] = 10000000
            else:
                phenotype_length = len(phenotype)
                # standard deviation
                stdev = numpy.std(projection[current_benchmark])
                fitness[current_benchmark] = 0.1*phenotype_length + 0.1*stdev
                for compare_benchmark in self.benchmarks:
                    if compare_benchmark == current_benchmark:
                        continue
                    else:                        
                        between_count = 0
                        collide_count = 0
                        element_count = len(projection[current_benchmark])
                        for i in xrange(len(projection[current_benchmark])):                            
                            # between max and min range of other projection
                            if max_projection[compare_benchmark]>projection[current_benchmark][i] and min_projection[compare_benchmark]<projection[current_benchmark][i]:
                                between_count += 1
                            # collide (not-separable)
                            for j in xrange(len(projection[compare_benchmark])):
                                if projection[compare_benchmark][j] == projection[current_benchmark][i]:
                                    collide_count += 1
                        fitness[current_benchmark] += 1*between_count + 100*collide_count
        return fitness

class Feature_Extractor(object):
    def __init__(self):
        self.training_records = []
        self.test_records = []
        self.variables = []
        self.measurement = 'unmatch_count' # unmatch_count or mse
        
    def process(self):
        # get classes
        training_records = self.training_records
        test_records = self.test_records
        variables = self.variables
        classes = []
        training_data = []
        training_target = []
        test_data = []
        test_target = []
        extractors = []
        for record in training_records:
            training_data.append(record[0:-1])
            training_target.append(record[-1])
            if record[-1] not in classes:
                classes.append(record[-1])
        for record in test_records:
            test_data.append(record[0:-1])
            test_target.append(record[-1])
        i = 0
        while len(variables)<(len(training_records[0])-1):
            variables.append('var_'+str(i))
            i+=1
        
        # GA SVM
        ga_svm = GA_SVM()
        ga_svm.training_data = training_data
        ga_svm.training_target = training_target
        ga_svm.label = 'GA SVM'
        ga_svm.stopping_value = 0
        extractors.append(ga_svm)
        
        # GE Multi-Fitness (My Hero :D )
        ge_multi_fitness = GE_Multi_Fitness()
        ge_multi_fitness.classes = classes
        ge_multi_fitness.variables = variables
        ge_multi_fitness.training_data = training_data
        ge_multi_fitness.training_target = training_target
        ge_multi_fitness.label = 'GE Multi Fitness'
        ge_multi_fitness.stopping_value = 0
        ge_multi_fitness.max_epoch = 100
        ge_multi_fitness.individual_length = 30
        ge_multi_fitness.population_size = 100
        extractors.append(ge_multi_fitness)
        
        # process extractors       
        for extractor in extractors:
            extractor.process()
                
        # show original svm performance
        show_svm(training_data, training_target, test_data, test_target, [], variables, 'Original SVM')
        
        # show ga_svm performance
        svc = ga_svm.best_individuals(1, benchmark='unmatch_count', representation='svc')
        gene = ga_svm.best_individuals(1, benchmark='unmatch_count', representation='default')
        show_svm(training_data, training_target, test_data, test_target, gene, variables, 'GA SVM', svc)
        
        # show ge_multi_fitness performance        
        # new features
        new_features = []
        for group in classes:
            best_phenotype = ge_multi_fitness.best_individuals(1, benchmark=group, representation='phenotype')
            if not (best_phenotype in new_features):
                new_features.append(best_phenotype)
        
        # training_data in new features
        new_training_data = []
        for record in training_data:
            new_record = []
            for feature in new_features:
                result, error = utils.execute(feature, record, variables)
                if error:
                    result = -1
                new_record.append(result)
            new_training_data.append(new_record)
        # test_data in new features
        new_test_data = []
        for record in test_data:
            new_record = []
            for feature in new_features:
                result, error = utils.execute(feature, record, variables)
                if error:
                    result = -1
                new_record.append(result)
            new_test_data.append(new_record)
        
        
        show_svm(new_training_data, training_target, new_test_data, test_target, [], new_features, 'GE Multi-Fitness')
        
        # show extractors graphics
        for extractor in extractors:
            extractor.show()
        
        # try hybrid
        '''
        ga_svm = GA_SVM()
        ga_svm.training_data = new_training_data
        ga_svm.training_target = training_target
        ga_svm.label = 'GA SVM'
        ga_svm.stopping_value = 0
        ga_svm.process()
        svc = ga_svm.best_individuals(1, benchmark='unmatch_count', representation='svc')
        gene = ga_svm.best_individuals(1, benchmark='unmatch_count', representation='default')
        show_svm(training_data, training_target, test_data, test_target, gene, variables, 'GA SVM', svc)
        ga_svm.show()
        '''

if __name__ == '__main__':
    
    
    # this is just for temporary, we will use iris dataset
    ds = datasets.load_iris()
    data = list(ds.data)
    target = list(ds.target)
    records = []
    for i in xrange(len(data)):
        record = []
        for j in xrange(len(data[i])):
            record.append(data[i][j])
        record.append(target[i])
        records.append(record)
    variables = ['petal_length','petal_width','sepal_length','sepal_width']
    training_records = records[0:20]+records[50:70]+records[130:150]
    
    
    randomizer = utils.randomizer
    records = []
    for i in xrange(200):
        x = randomizer.randrange(-7,7)
        y = randomizer.randrange(-7,7)
        r = (x**2+y**2)
        if r<3:
            c = 0
        elif r<6:
            c = 1
        else:
            c = 2
        records.append([x,y,c])
    training_records = records[0:50]
    variables = ['x','y']    
    
    
    # make feature extractor
    fe = Feature_Extractor()
    fe.training_records = training_records
    fe.test_records = records
    fe.variables = variables
    fe.measurement = 'error'
    fe.process()
