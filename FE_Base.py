'''
Created on Nov 13, 2012

@author: gofrendi
'''
import os, sys
lib_path = os.path.abspath('./gogenpy')
sys.path.insert(0,lib_path)

import numpy, time
import math
from gogenpy import classes, utils
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter

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
    num_types = (float,int,long,complex)
    if (len(data_1) == len(data_2)) and len(data_1)>0:
        se = 0.0
        for i in xrange(len(data_1)):
            # both data is number
            if isinstance(data_1[i], num_types) and isinstance(data_2[i], num_types):
                se += (data_1[i] - data_2[i]) ** 2
            else:
                se += 1
        mse = se/len(data_1)
        return mse
    else:
        raise TypeError

def get_result(data_1, data_2):
    unmatch = count_unmatch(data_1,data_2)
    mse = calculate_mse(data_1,data_2)
    data_count = len(data_1)
    error_percentage = (float(unmatch)/float(data_count)) * 100.0
    accuracy = 100.0 - error_percentage
    return {"unmatch":unmatch, "mse":mse, "data_count":data_count, "error_percentage":error_percentage, "accuracy":accuracy}

def gene_to_feature(data, mask):
    new_data = []
    if type(data[0]) is list:
        for i in xrange(len(data)):
            new_record = []
            for j in xrange(6, len(data[i])+6):
                if mask[j] == '1':
                    new_record.append(data[i][j-6])
            new_data.append(new_record)
    else:
        for i in xrange(6, len(data)+6):
            if mask[i] == '1':
                new_data.append(data[i-6])
    return new_data

def gene_to_svm(gene):
    kernel_gene = gene[0:2]
    degree_gene = gene[2:4]
    gamma_gene = gene[4:6]
    kernel_option = {'00':'linear', '01':'linear', '10':'rbf', '11':'poly'}
    degree_value = utils.bin_to_dec(degree_gene)+1
    gamma_value = float(utils.bin_to_dec(gamma_gene)+1)/10.0
    svc = None
    try:
        # sklearn 0.13 can use max-iter
        svc = svm.SVC(kernel=kernel_option[kernel_gene], degree=degree_value, gamma=gamma_value, class_weight='auto', max_iter=1000)
    except:
        # sklearn 0.12.1 can't use max-iter
        svc = svm.SVC(kernel=kernel_option[kernel_gene], degree=degree_value, gamma=gamma_value, class_weight='auto')
    return svc

def get_svm_result(training_data, training_target, test_data, test_target, old_features=[], new_features=[], label='', svc=None):
    utils.write("Processing SVM '%s'" % (label))
    
    if svc is None:
        try:
            svc = svm.SVC(max_iter=1000)
        except:
            svc = svm.SVC()
    
    start_time = time.time()
    new_training_data = build_new_data(training_data, old_features, new_features)
    end_time = time.time()
    training_preprocessing_time = end_time - start_time
    
    start_time = time.time()
    new_test_data = build_new_data(test_data, old_features, new_features)
    end_time = time.time()
    test_preprocessing_time = end_time - start_time
        
    utils.write("Training SVM '%s'" % (label))
    start_time = time.time()
    svc.fit(new_training_data, training_target)    
    end_time = time.time()
    utils.write("Done training SVM '%s'" % (label))
    training_time = end_time - start_time
    
    utils.write("Executing SVM '%s'" % (label))
    start_time = time.time()    
    svc_training_prediction = svc.predict(new_training_data)
    svc_test_prediction = svc.predict(new_test_data)
    end_time = time.time()
    utils.write("Done executing '%s'" % (label))
    execution_time = end_time - start_time
    utils.write("Done Processing SVM '%s'" % (label))
    print('')
    
    training_result = get_result(training_target, svc_training_prediction)
    test_result = get_result(test_target, svc_test_prediction)
        
    label = ' '+label+' '
    limitter_count = int((70-len(label))/2)
    result = ''
    result += '='*limitter_count + label + '='*limitter_count+'\r\n'
    result += 'TRAINING MSE      : '+str(training_result['mse'])+'\r\n'
    result += 'TRAINING UNMATCH  : '+str(training_result['unmatch'])+' of '+str(training_result['data_count'])+' ('+str(training_result['error_percentage'])+'%)\r\n'
    result += 'TRAINING ACCURACY : '+str(training_result['accuracy'])+'%\r\n'
    result += 'TEST MSE          : '+str(test_result['mse'])+'\r\n'
    result += 'TEST UNMATCH      : '+str(test_result['unmatch'])+' of '+str(test_result['data_count'])+' ('+str(test_result['error_percentage'])+'%)\r\n'
    result += 'TEST ACCURACY     : '+str(test_result['accuracy'])+'%\r\n'
    result += 'FEATURES COUNT    : '+str(len(new_features))+'\r\n'
    result += 'USED FEATURES     : '+", ".join(new_features)+'\r\n'    
    result += 'KERNEL            : '+svc.kernel+'\r\n'
    if svc.kernel=='poly':
        result += 'DEGREE            : '+str(svc.degree)+'\r\n'
    elif svc.kernel=='rbf':
        result += 'GAMMA             : '+str(svc.gamma)+'\r\n'
    result += 'TRAINING PREP     : '+str(training_preprocessing_time)+' second(s)\r\n'
    result += 'TEST PREP         : '+str(test_preprocessing_time)+' second(s)\r\n'
    result += 'TRAINING TIME     : '+str(training_time)+' second(s)\r\n'
    result += 'TEST TIME         : '+str(execution_time)+' second(s)\r\n\r\n'
    
    # delete svc after use it
    del svc  
    
    return {"str":result, "training_result":training_result, "test_result":test_result}

class Draw_projection_y_formatter(Formatter):
    def __init__(self, groups):
        self.groups = groups
    
    def __call__(self, y, pos=0):
        'Return the label for time x at position pos'
        if y in xrange(len(self.groups)):
            return self.groups[int(y)]
        elif y == len(self.groups):
            return 'global'
        else:
            return ''

def draw_projection(data, targets, old_features, new_features, label=''):
    
    groups = []
    for target in targets:
        if not(target in groups):
            groups.append(target)
    new_feature_count = len(new_features)
    group_count = len(groups)
    window_per_row = 1;
    window_per_col = 1;
    if new_feature_count<2:
        window_per_col = new_feature_count
        window_per_row = 1
    else:
        window_per_col = 2
        if(new_feature_count%window_per_col)>0:
            window_per_row = new_feature_count/window_per_col + 1
        else:
            window_per_row = new_feature_count/window_per_col
    
    formatter = Draw_projection_y_formatter(groups)
        
    fig = plt.figure(figsize=(20.0, 12.0))
    for feature_index in xrange(new_feature_count):
        sp =fig.add_subplot(window_per_row, window_per_col, feature_index+1)             
        new_feature = new_features[feature_index]         
        error, global_projection, projection, min_projection, max_projection = calculate_projection(data, targets, old_features, new_feature)
        if error:
            continue
        group_index = 0
        for group in groups:
            local_projection = projection[group]
            for value in local_projection:
                sp.plot(value,group_index, 'bo')
            group_index += 1
        for value in global_projection:
            sp.plot(value,group_index, 'bo')
        del(min_projection)
        del(max_projection)
        sp.set_title('Feature: '+new_feature)
        sp.set_ylabel('Classes')
        sp.set_xlabel('Projection')
        sp.set_ylim(-0.1, group_count+0.1)
        sp.set_xlim(min(global_projection)-0.1, max(global_projection)+0.1)
        sp.yaxis.set_major_formatter(formatter)
    plt.suptitle('Feature Projection '+label)
    plt.savefig('Feature Projection '+label, dpi=100)
    del formatter
    del fig

def calculate_projection(data, targets, old_features, new_feature):
    '''
    return error, global projection, per-group projection, per-group max projection value and per-group min projection value
    when training_data and training_target projected by the phenotype
    '''
    error = False
    groups = []
    for target in targets:
        if not (target in groups):
            groups.append(target)
    projection = {}
    global_projection = []
    min_projection = {}
    max_projection = {}
    for group in groups:
        projection[group] = []
        min_projection[group] = 0
        max_projection[group] = 0
        
    # calculate projection
    for i in xrange(len(data)):
        record = data[i]
        for group in groups:
            if group == targets[i]:
                result, error = utils.execute(new_feature, record, old_features)
                if error:
                    return error, global_projection, projection, min_projection, max_projection
                else:
                    result = float(result)
                    projection[group].append(result)
                    global_projection.append(result)
    for group in groups:
        min_projection[group] = min(projection[group])
        max_projection[group] = max(projection[group])
    return error, global_projection, projection, min_projection, max_projection

def build_new_data(data, old_features, new_features):
    '''
    This function is used to represent data in new_features
    '''
    new_data = []
    for record in data:
        new_record = []
        for feature in new_features:
            result, error = utils.execute(feature, record, old_features)
            if error:
                result = -1
            new_record.append(result)
        new_data.append(new_record)
    return new_data

class GA_SVM(classes.Genetics_Algorithm): 
    def __init__(self):
        super(GA_SVM,self).__init__()
        self.fitness_measurement = 'MIN'
        self.representations=['default','svc', 'unmatch_count']
        self.benchmarks=['unmatch_count', 'mse']
        self.individual_length=20
        self.training_data = []
        self.training_num_target = []
    
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
            # feature selection by using first digits of the gene
            new_training_data = gene_to_feature(self.training_data, gene)
            # perform svm
            svc = gene_to_svm(gene)
            svc.fit(new_training_data, self.training_num_target)
            # predict
            prediction = svc.predict(new_training_data)
            individual['svc'] = svc
            individual['unmatch_count'] = count_unmatch(self.training_num_target, prediction)
            individual['mse'] = calculate_mse(self.training_num_target, prediction)
        return individual
           
    def do_calculate_fitness(self, individual):                    
        return {'unmatch_count': individual['unmatch_count'], 'mse':individual['mse']}

class GE_Base(classes.Grammatical_Evolution):
    def __init__(self):
        super(GE_Base, self).__init__()
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
    
    def _bad_fitness(self):
        fitness = {}
        for benchmark in self.benchmarks:
            fitness[benchmark] = 1000000000
        return fitness
    
    def _calculate_projection(self, phenotype):
        '''
        return error, global projection, per-group projection, per-group max projection value and per-group min projection value
        when training_data and training_target projected by the phenotype
        '''
        data = self.training_data
        targets = self.training_target
        old_features = self.variables        
        return calculate_projection(data, targets, old_features, phenotype)
    
    def _pack_projection_attribute(self, global_stdev, phenotype_complexity, local_stdev, between_count, collide_count, 
                                   projection_count, intersection_range, projection_range):
        return {
           'global_stdev' : global_stdev,
           'phenotype_complexity' : phenotype_complexity,
           'local_stdev' : local_stdev,
           'between_count' : between_count,
           'collide_count' : collide_count,
           'projection_count' : projection_count,
           'intersection_range' : intersection_range,
           'projection_range' : projection_range
        }
    
    def _calculate_projection_attribute(self, phenotype):
        '''
        return a dictionary which contains of global_stdev, 
        phenotype_complexity, local_stdev, between_count, collide_count and projection_count
        '''
        
        # calculate projection
        start_time = time.time()        
        error, global_projection, projection, max_projection, min_projection = self._calculate_projection(phenotype)
        end_time = time.time()
        time_complexity = end_time - start_time
        
        global_stdev = 0.0000000001 # avoid division by zero
        phenotype_complexity = time_complexity
        local_stdev = {}
        between_count = {}
        collide_count = {}
        projection_count = {}
        intersection_range = {}
        projection_range = 0.0000000001
        
        
        if error:
            attributes = self._pack_projection_attribute(global_stdev, phenotype_complexity,
                local_stdev, between_count, collide_count, projection_count, intersection_range, projection_range)
            return error, attributes
        
        # calculate projection attribute                    
        global_stdev = max(numpy.std(global_projection), 0.0000000001) # avoid division by zero
        projection_range = max(max(global_projection) - min(global_projection), 0.0000000001)
        for current_group in self.classes:
            # stdev
            local_stdev[current_group] = numpy.std(projection[current_group])
            between_count[current_group] = 0.0
            collide_count[current_group] = 0.0
            intersection_range[current_group] = 0.0
            projection_count[current_group] = float(len(projection[current_group]))
            for compare_group in self.classes:
                if compare_group == current_group:
                    continue
                else:
                    max_min = max(min(projection[current_group]),min(projection[compare_group]))
                    min_max = min(max(projection[current_group]),max(projection[compare_group]))
                    if (min_max - max_min) > 0:
                        intersection_range[current_group] += (min_max - max_min)
                    for i in xrange(len(projection[current_group])):                            
                        # between max and min range of other projection
                        if max_projection[compare_group]>projection[current_group][i] and min_projection[compare_group]<projection[current_group][i]:
                            between_count[current_group] += 1
                        # collide (not-separable)
                        for j in xrange(len(projection[compare_group])):
                            if projection[compare_group][j] == projection[current_group][i]:
                                collide_count[current_group] += 1
        # return projection attributes
        attributes = self._pack_projection_attribute(global_stdev, phenotype_complexity, 
            local_stdev, between_count, collide_count, projection_count, intersection_range, projection_range)
        return error, attributes
    
    def process(self):
        self.grammar['<var>'] = self.variables        
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
        
class GE_Multi_Fitness(GE_Base):    
    def __init__(self):
        super(GE_Multi_Fitness, self).__init__()
    
    def process(self):
        # adjust benchmark        
        self.benchmarks = self.classes
        GE_Base.process(self)
        
    def do_calculate_fitness(self, individual):
        # calculate projection attribute
        phenotype = individual['phenotype']
        error, attributes = self._calculate_projection_attribute(phenotype)        
        if error:
            return self._bad_fitness()
        
        global_stdev = attributes['global_stdev']
        phenotype_complexity = attributes['phenotype_complexity']
        local_stdev = attributes['local_stdev']
        between_count = attributes['between_count']
        collide_count = attributes['collide_count']
        projection_count = attributes['projection_count']
        intersection_range = attributes['intersection_range']
        projection_range = attributes['projection_range']
        
        # calculate fitness
        fitness = self._bad_fitness()        
        for group in self.classes:            
            fitness[group] = 0.0 * local_stdev[group]/global_stdev + \
                phenotype_complexity + \
                (10 * between_count[group]/projection_count[group]) + \
                (100 * collide_count[group]/projection_count[group]) + \
                (100 * intersection_range[group]/projection_range)            
        return fitness

class GE_Global_Fitness(GE_Base):
    def __init__(self):
        super(GE_Global_Fitness, self).__init__()
    
    def _bad_fitness(self):
        return {'default':100000000}
    
    def do_calculate_fitness(self, individual):
        # calculate projection attribute
        phenotype = individual['phenotype']
        error, attributes = self._calculate_projection_attribute(phenotype)        
        if error:
            return self._bad_fitness()
        
        global_stdev = attributes['global_stdev']
        phenotype_complexity = attributes['phenotype_complexity']
        local_stdev = attributes['local_stdev']
        between_count = attributes['between_count']
        collide_count = attributes['collide_count']
        projection_count = attributes['projection_count']
        intersection_range = attributes['intersection_range']
        projection_range = attributes['projection_range']
        
        # calculate fitness
        bad_accumulation = 0
        for group in self.classes:
            bad_accumulation += 0.0 * local_stdev[group]/global_stdev +\
                phenotype_complexity + \
                (10 * between_count[group]/projection_count[group]) + \
                (100 * collide_count[group]/projection_count[group]) + \
                (100 * intersection_range[group]/projection_range) 
        fitness_value = phenotype_complexity+ bad_accumulation/len(self.classes)
        # return fitness value
        fitness = {}
        fitness['default'] = fitness_value
        return fitness

class Feature_Extractor(object):
    def __init__(self):
        self.records = []
        self.fold = 5
        self.variables = []
        self.measurement = 'unmatch_count' # unmatch_count or mse
        self.max_epoch = 100
        self._target_dict = {}
        self.label = ''
        self.population_size = 100
        
    def process(self):
        # prepare variables        
        data = []
        labels = []
        label_targets = []
        num_targets = []
        label_records = []
        num_records = []
        target_dictionary = {}
        target_indexes = {}
        training_count_per_fold = {}
        # fill out the variables
        class_index = 0
        i = 0
        for record in self.records:
            data.append(record[:-1])
            label_targets.append(record[-1])
            if not (record[-1] in target_dictionary):
                target_dictionary[record[-1]] = class_index
                target_indexes[record[-1]] = []
                labels.append(record[-1])
                class_index += 1            
            num_targets.append(target_dictionary[record[-1]])
            label_records.append(record[:-1]+[record[-1]])
            num_records.append(record[:-1]+[target_dictionary[record[-1]]])
            # add target indexes
            target_indexes[record[-1]].append(i)
            i+=1
        for label in label_targets:
            training_count_per_fold[label] = math.floor(len(target_indexes[label])/self.fold)
        
        # folding scenario
        output = ''
        original_svm_result = []
        ga_svm_result = []
        ge_global_fitness_result = []
        ge_multi_fitness_result = []
        
        variables = self.variables
        for fold_index in xrange(self.fold):            
            training_indexes = []
            for label in label_targets:
                for i in xrange(int(fold_index*training_count_per_fold[label]), int((fold_index+1)*training_count_per_fold[label])):
                    training_indexes.append(target_indexes[label][i])
            training_data = []            
            training_label_targets = []
            training_num_targets = []
            test_data = []
            test_label_targets = []
            test_num_targets = []
            for i in(xrange(len(data))):
                if i in training_indexes:
                    training_data.append(data[i])
                    training_label_targets.append(label_targets[i])
                    training_num_targets.append(num_targets[i])
                else:
                    test_data.append(data[i])
                    test_label_targets.append(label_targets[i])
                    test_num_targets.append(num_targets[i])
            
            print('FOLD : '+str(fold_index+1))
            
            
            # Original SVM
            svm_result = get_svm_result(training_data, training_num_targets, test_data, test_num_targets, variables, variables, self.label+' Original SVM Fold '+str(fold_index+1))
            output += svm_result['str']
            original_svm_result.append(svm_result)
            draw_projection(training_data, training_label_targets, variables, variables, ' Original SVM "'+self.label+'" Fold '+str(fold_index+1)+' Training')
            draw_projection(test_data, test_label_targets, variables, variables, ' Original SVM "'+self.label+'" Fold '+str(fold_index+1)+' Test')
            
            # GA SVM
            ga_svm = GA_SVM()
            ga_svm.training_data = training_data
            ga_svm.training_num_target = training_num_targets
            ga_svm.label = 'GA SVM "'+self.label+'" Fold '+str(fold_index+1)
            ga_svm.stopping_value = 0
            ga_svm.max_epoch = self.max_epoch
            ga_svm.population_size = self.population_size
            ga_svm.process()
            gene = ga_svm.best_individuals(1, benchmark='unmatch_count', representation='default')
            new_variables = gene_to_feature(variables, gene)
            svm_result = get_svm_result(training_data, training_num_targets, test_data, test_num_targets, variables, new_variables, ga_svm.label, gene_to_svm(gene))
            output += svm_result['str']
            ga_svm_result.append(svm_result)
            ga_svm.show(True, ga_svm.label+'.png')
            draw_projection(training_data, training_label_targets, variables, new_variables, ga_svm.label+' Training')
            draw_projection(test_data, test_label_targets, variables, new_variables, ga_svm.label+' Test')
            
            del ga_svm
            
            # GE Global-Fitness (My Previous Research)
            ge_global_fitness = GE_Global_Fitness()
            ge_global_fitness.classes = labels
            ge_global_fitness.variables = variables
            ge_global_fitness.training_data = training_data
            ge_global_fitness.training_target = training_label_targets
            ge_global_fitness.label = 'GE Global Fitness "'+self.label+'" Fold '+str(fold_index+1)
            ge_global_fitness.stopping_value = 0.1
            ge_global_fitness.max_epoch = self.max_epoch
            ge_global_fitness.individual_length = 30
            ge_global_fitness.population_size = self.population_size
            ge_global_fitness.process()
            # new features
            best_phenotypes = ge_global_fitness.best_individuals(len(labels), representation='phenotype')
            new_features = []
            for best_phenotype in best_phenotypes:
                if not (best_phenotype in new_features):
                    new_features.append(best_phenotype)
            try:
                svc = svm.SVC(max_iter=1000)
            except:
                svc = svm.SVC()
            svm_result = get_svm_result(training_data, training_num_targets, test_data, test_num_targets, variables, new_features, ge_global_fitness.label, svc)
            output += svm_result['str']
            ge_global_fitness_result.append(svm_result)
            ge_global_fitness.show(True, ge_global_fitness.label+'.png')
            draw_projection(training_data, training_label_targets, variables, new_features, ge_global_fitness.label+' Training')
            draw_projection(test_data, test_label_targets, variables, new_features, ge_global_fitness.label+' Test')
            
            del ge_global_fitness
            
            # GE Multi-Fitness (My Hero :D )
            ge_multi_fitness = GE_Multi_Fitness()
            ge_multi_fitness.classes = labels
            ge_multi_fitness.variables = variables
            ge_multi_fitness.training_data = training_data
            ge_multi_fitness.training_target = training_label_targets
            ge_multi_fitness.label = 'GE Multi Fitness "'+self.label+'" Fold '+str(fold_index+1)
            ge_multi_fitness.stopping_value = 0.1
            ge_multi_fitness.max_epoch = self.max_epoch
            ge_multi_fitness.individual_length = 30
            ge_multi_fitness.population_size = self.population_size
            ge_multi_fitness.process()
            # new features
            new_features = []
            for group in labels:
                best_phenotype = ge_multi_fitness.best_individuals(1, benchmark=group, representation='phenotype')
                if not (best_phenotype in new_features):
                    new_features.append(best_phenotype)
            try:
                svc = svm.SVC(max_iter=1000)
            except:
                svc = svm.SVC()
            svm_result = get_svm_result(training_data, training_num_targets, test_data, test_num_targets, variables, new_features, ge_multi_fitness.label, svc)
            output += svm_result['str']
            ge_multi_fitness_result.append(svm_result)
            ge_multi_fitness.show(True, ge_multi_fitness.label+'.png')
            draw_projection(training_data, training_label_targets, variables, new_features, ge_multi_fitness.label+' Training')
            draw_projection(test_data, test_label_targets, variables, new_features, ge_multi_fitness.label+' Test')
            
            del ge_multi_fitness
            
        print output
        
        text_file = open('SVM training and test comparison '+self.label+'.txt', "w")
        text_file.write(output)
        text_file.close()
        
        fig = plt.figure(figsize=(20.0, 12.0))
        sp_1 = fig.add_subplot(1, 2, 1)
        sp_2 = fig.add_subplot(1, 2, 2)
        
        original_svm_training_accuracy = []
        original_svm_test_accuracy = []
        ga_svm_training_accuracy = []
        ga_svm_test_accuracy = []
        ge_global_fitness_training_accuracy = []
        ge_global_fitness_test_accuracy = []
        ge_multi_fitness_training_accuracy = []
        ge_multi_fitness_test_accuracy = []
        for i in xrange(self.fold):
            original_svm_training_accuracy.append(original_svm_result[i]['training_result']['accuracy'])
            original_svm_test_accuracy.append(original_svm_result[i]['test_result']['accuracy'])
            ga_svm_training_accuracy.append(ga_svm_result[i]['training_result']['accuracy'])
            ga_svm_test_accuracy.append(ga_svm_result[i]['test_result']['accuracy'])
            ge_global_fitness_training_accuracy.append(ge_global_fitness_result[i]['training_result']['accuracy'])
            ge_global_fitness_test_accuracy.append(ge_global_fitness_result[i]['test_result']['accuracy'])
            ge_multi_fitness_training_accuracy.append(ge_multi_fitness_result[i]['training_result']['accuracy'])
            ge_multi_fitness_test_accuracy.append(ge_multi_fitness_result[i]['test_result']['accuracy'])
        fold_indexes = list(xrange(self.fold))
        sp_1.plot(fold_indexes, original_svm_training_accuracy, label="SVM")
        sp_2.plot(fold_indexes, original_svm_test_accuracy, label="SVM")
        sp_1.plot(fold_indexes, ga_svm_training_accuracy, label="GA SVM")
        sp_2.plot(fold_indexes, ga_svm_test_accuracy, label="GA SVM")
        sp_1.plot(fold_indexes, ge_global_fitness_training_accuracy, label="GE Global")
        sp_2.plot(fold_indexes, ge_global_fitness_test_accuracy, label="GE Global")
        sp_1.plot(fold_indexes, ge_multi_fitness_training_accuracy, label="GE Multi")
        sp_2.plot(fold_indexes, ge_multi_fitness_test_accuracy, label="GE Multi")
        
        sp_1.set_title('Training Accuration')
        sp_1.set_ylabel('Accuration (%)')
        sp_1.set_xlabel('Fold')
        sp_1.set_ylim(-0.5,100.5)
        sp_1.legend(shadow=True, loc=0)
        
        sp_2.set_title('Test Accuration')
        sp_2.set_ylabel('Accuration (%)')
        sp_2.set_xlabel('Fold')
        sp_2.set_ylim(-0.5,100.5)
        sp_2.legend(shadow=True, loc=0)
        
        plt.suptitle('SVM training and test comparison of '+self.label)

        plt.savefig('SVM training and test comparison "'+self.label+'".png', dpi=100)
        del fig