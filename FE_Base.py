'''
Created on Nov 13, 2012

@author: gofrendi
'''
import os, sys, gc, shutil
lib_path = os.path.abspath('./gogenpy')
sys.path.insert(0,lib_path)

import csv, time
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

def get_svm_result(training_data, training_target, test_data, test_target, old_features=[], new_features=[], label='', svc=None):
    utils.write("Processing SVM '%s'" % (label))
    
    if svc is None:
        try:
            svc = svm.SVC(max_iter=2000, class_weight='auto')
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

class Draw_accuracy_x_formatter(Formatter):
    def __init__(self, fold_count):
        self.fold_count = fold_count
    def __call__(self, x, pos=0):
        if (x-int(x) == 0) and (x<self.fold_count):
            return 'Fold '+str(int(x)+1)
        else:
            return ''

def calculate_histogram(array):
    histogram = {}
    for value in array:
        if not value in histogram:
            histogram[value] = 0.0
        histogram[value]+=1
    return histogram

def draw_projection(data, targets, old_features, new_features, plot_label='', file_name=''):
    
    groups = []
    for target in targets:
        if not(target in groups):
            groups.append(target)
    groups.sort()
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
    
    y_formatter = Draw_projection_y_formatter(groups)    
        
    fig = plt.figure(figsize=(20.0, 12.0))
    for feature_index in xrange(new_feature_count):
        sp =fig.add_subplot(window_per_row, window_per_col, feature_index+1)             
        new_feature = new_features[feature_index]         
        error, projection = calculate_projection(data, targets, old_features, new_feature)
        min_projection = {}
        max_projection = {}
        global_projection = []
        for label in projection:
            global_projection += projection[label]
            if len(projection[label])>0:
                min_projection[label] = min(projection[label])
                max_projection[label] = max(projection[label])
            else:
                min_projection[label] = 0
                max_projection[label] = 0
        
        if error:
            continue
        group_index = 0
        for group in groups:
            sp.plot([min_projection[group],max_projection[group]], [group_index, group_index], 'b--')
            sp.plot([min_projection[group],min_projection[group]], [group_index, group_count], 'b--')
            sp.plot([max_projection[group],max_projection[group]], [group_index, group_count], 'b--')
            local_projection = projection[group]
                
            histogram = calculate_histogram(local_projection)
            max_histogram_value = 0.01
            # get maximum histogram value
            for value in histogram:
                if histogram[value]>max_histogram_value:
                    max_histogram_value = histogram[value]
            for value in histogram:
                count = histogram[value]
                normalized_count = 0.8 * count/max_histogram_value
                sp.plot([value, value], [group_index, group_index+normalized_count], color='k', linewidth=4)
                sp.plot(value,len(groups), 'bo')
            group_index += 1
                    
        sp.set_title('Feature: '+new_feature)
        sp.set_ylabel('Classes')
        sp.set_xlabel('Projection')
        y_range = group_count
        sp.set_ylim(0-0.1*y_range, group_count+0.1*y_range)
        x_range = max(global_projection) - min(global_projection)
        sp.set_xlim(min(global_projection)-0.1*x_range, max(global_projection)+0.1*x_range)
        sp.yaxis.set_major_formatter(y_formatter)
    plt.suptitle('Feature Projection '+plot_label)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    plt.savefig(file_name, dpi=100)
    
    fig.clf()
    plt.close()
    gc.collect()

def calculate_projection(data, targets, old_features, new_feature):
    '''
    return error, global projection, per-group projection, per-group max projection value and per-group min projection value
    when training_data and training_target projected by the phenotype
    '''
    error = False
    groups = []
    global_projection = []
    for target in targets:
        if not (target in groups):
            groups.append(target)
    projection = {}
    for group in groups:
        projection[group] = []
        
    # calculate projection
    for i in xrange(len(data)):
        record = data[i]
        for group in groups:
            if group == targets[i]:
                result, error = utils.execute(new_feature, record, old_features)
                if error:
                    return error, projection
                else:
                    result = float(result)
                    projection[group].append(result)
                    global_projection.append(result)
    # normalization
    max_value = max(global_projection)
    min_value = min(global_projection)
    projection_range = max_value - min_value
    if projection_range == 0:
        projection_range = 1
    for group in groups:
        for i in xrange(len(projection[group])):
            projection[group][i] /= projection_range
    
    return error, projection

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

class SVM_Preprocessor(object):
    def get_new_features(self):
        return []
    
    def get_svm(self):
        svc = None
        try:
            svc = svm.SVC(max_iter=2000, class_weight='auto')
        except:
            svc = svm.SVC()
        return svc 

class GA_SVM(classes.Genetics_Algorithm, SVM_Preprocessor): 
    def __init__(self):
        super(GA_SVM,self).__init__()
        self.fitness_measurement = 'MIN'
        self.representations=['default','svc', 'unmatch_count']
        self.benchmarks=['unmatch_count', 'mse']
        self.individual_length=20
        self.training_data = []
        self.training_num_target = []
        self.classes = []
        self.variables = []
    
    def process(self):
        if not len(self.training_data) == 0 and not len(self.training_data[0]) == 0:
            self.individual_length = len(self.variables)+6
        else:
            self.individual_length = 20
        classes.Genetics_Algorithm.process(self)
    
    def do_process_individual(self, individual):
        gene = individual['default']
        # if the dataset is empty, then everything is impossible, don't try to do anything
        if not len(self.training_data) == 0 and not len(self.training_data[0]) == 0:           
            # feature selection by using first digits of the gene
            new_features = self._gene_to_feature(gene)
            new_training_data = build_new_data(self.training_data, self.variables, new_features)            
            # perform svm
            svc = self._gene_to_svm(gene)
            svc.fit(new_training_data, self.training_num_target)
            # predict
            prediction = svc.predict(new_training_data)
            individual['svc'] = svc
            individual['unmatch_count'] = count_unmatch(self.training_num_target, prediction)
            individual['mse'] = calculate_mse(self.training_num_target, prediction)
        return individual
    
    def _gene_to_feature(self, mask):
        new_features = []
        for i in xrange(6, len(self.variables)+6):
            if mask[i] == '1':
                new_features.append(self.variables[i-6])
        return new_features
    
    def _gene_to_svm(self, gene):
        kernel_gene = gene[0:2]
        degree_gene = gene[2:4]
        gamma_gene = gene[4:6]
        kernel_option = {'00':'linear', '01':'linear', '10':'rbf', '11':'poly'}
        degree_value = utils.bin_to_dec(degree_gene)+1
        gamma_value = float(utils.bin_to_dec(gamma_gene)+1)/10.0
        svc = None
        try:
            # sklearn 0.13 can use max-iter
            svc = svm.SVC(kernel=kernel_option[kernel_gene], degree=degree_value, gamma=gamma_value, class_weight='auto', max_iter=2000)
        except:
            # sklearn 0.12.1 can't use max-iter
            svc = svm.SVC(kernel=kernel_option[kernel_gene], degree=degree_value, gamma=gamma_value, class_weight='auto')
        return svc
               
    def do_calculate_fitness(self, individual):                    
        return {'unmatch_count': individual['unmatch_count'], 'mse':individual['mse']}
    
    def get_new_features(self):
        gene = self.best_individuals(1, 'unmatch_count', 'default')
        return self._gene_to_feature(gene)
    
    def get_svm(self):
        gene = self.best_individuals(1, 'unmatch_count', 'default')
        return self._gene_to_svm(gene)

class GE_Base(classes.Grammatical_Evolution, SVM_Preprocessor):
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
    
    def get_num_targets(self):
        num_targets = []
        target_dictionary = {}
        # fill out the variables
        class_index = 0
        for target in self.training_target:
            if not (target in target_dictionary):
                target_dictionary[target] = class_index
                class_index += 1            
            num_targets.append(target_dictionary[target])
        return num_targets
    
    def _bad_fitness(self):
        fitness = {}
        for benchmark in self.benchmarks:
            fitness[benchmark] = 1000000
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
    
    def _calculate_projection_attribute(self, phenotype):
        '''
        return a dictionary which contains of important attributes
        '''
        
        # calculate projection
        start_time = time.time()        
        error, projection = self._calculate_projection(phenotype)
        min_projection = {}
        max_projection = {}
        global_projection = []
        for label in projection:
            global_projection += projection[label]
            if(len(projection[label])>0):
                min_projection[label] = min(projection[label])
                max_projection[label] = max(projection[label])
            else:
                min_projection[label] = 0
                max_projection[label] = 0
        end_time = time.time()
        time_complexity = end_time - start_time
        
        neighbour_distances = {}
        intrusion_damages = {}
        collision_damages = {}
        surrounded_damages = {}
        
        try:
            
            if error:
                raise Exception("Error")
            
            histogram = {}
            for group in self.classes:
                histogram[group] = calculate_histogram(projection[group])
                surrounded_damages[group] = 0.0
            
            for current_group in self.classes:
                current_histogram = histogram[current_group]
                #current_projection_count = len(projection[current_group])
                
                # neighbour_distance & intruder
                current_max = max_projection[current_group]
                current_min = min_projection[current_group]
                current_range = max(current_max-current_min, 0.0001) # avoid division by zero later
                neighbour_distance = 1.0 # since it's normalized it's save to assume maximum distance is equal to 1
                intrusion_damage = 0.0
                collision_damage = 0.0
                for compare_group in self.classes:
                    if compare_group == current_group:
                        continue
                    # neighbour_distance
                    compare_max = max_projection[compare_group]
                    compare_min = min_projection[compare_group]
                    new_neighbour_distance = min(
                            abs(current_max - compare_min),
                            abs(compare_max - current_min)
                        )
                    if new_neighbour_distance<neighbour_distance:
                        neighbour_distance = new_neighbour_distance
                    # intruder
                    compare_histogram = histogram[compare_group]
                    for compare_value in compare_histogram:
                        if compare_value<=current_max and compare_value>=current_min:
                            intrusion_distance = min(
                                            current_max-compare_value,
                                            compare_value-current_min
                                    )
                            intrusion_distance = max(intrusion_distance, 0.0001)
                            intruder_count = compare_histogram[compare_value]
                            # intrusion_damage += (intrusion_distance * intruder_count) / (current_range * current_projection_count)
                            intrusion_damage += (intrusion_distance * intruder_count) / current_range
                            surrounded_damages[compare_group] += (intrusion_distance * intruder_count) /current_range
                        for current_value in current_histogram:
                            if compare_value == current_value:
                                collision_count = compare_histogram[compare_value] + current_histogram[current_value]
                                # collision_damage = collision_count/current_projection_count
                                collision_damage = collision_count
                        
                neighbour_distances[current_group] = neighbour_distance
                intrusion_damages[current_group] = intrusion_damage
                collision_damages[current_group] = collision_damage
                    
        except:
            error = True

        # return projection attributes
        attributes = {
           'neighbour_distance' : neighbour_distances,
           'intrusion_damage' : intrusion_damages,
           'surrounded_damage' : surrounded_damages,
           'collision_damage' : collision_damages,
           'time_complexity' : time_complexity
        }
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
        
        neighbour_distances = attributes['neighbour_distance']
        intrusion_damages = attributes['intrusion_damage']
        surrounded_damages = attributes['surrounded_damage']
        collision_damages = attributes['collision_damage']
        time_complexity = attributes['time_complexity']
        
        # calculate fitness
        fitness = self._bad_fitness()        
        for group in self.classes:
            try:
                fitness[group] = \
                    (10 * time_complexity) + \
                    (1/(100 * neighbour_distances[group])) + \
                    (100 * intrusion_damages[group]) + \
                    (10 * surrounded_damages[group]) + \
                    (1000 * collision_damages[group])
            except:
                return self._bad_fitness()
        return fitness
    
    def get_new_features(self):
        new_features = []
        for group in self.classes:
            best_phenotype = self.best_individuals(1, benchmark=group, representation='phenotype')
            if not (best_phenotype in new_features):
                new_features.append(best_phenotype)
        return new_features
    
    def get_svm(self):
        svc = None
        try:
            svc = svm.SVC(max_iter=2000, class_weight='auto')
        except:
            svc = svm.SVC()
        return svc 

class GE_Global_Fitness(GE_Base):
    def __init__(self):
        super(GE_Global_Fitness, self).__init__()
    
    def do_calculate_fitness(self, individual):
        # calculate projection attribute
        phenotype = individual['phenotype']
        error, attributes = self._calculate_projection_attribute(phenotype)        
        if error:
            return self._bad_fitness()
        
        # local_projection = attributes['local_projection']
        neighbour_distances = attributes['neighbour_distance']
        intrusion_damages = attributes['intrusion_damage']
        surrounded_damages = attributes['surrounded_damage']
        collision_damages = attributes['collision_damage']
        time_complexity = attributes['time_complexity']
        
        # calculate fitness
        try:
            bad_accumulation = 0
            for group in self.classes:
                # local_projection_count = len(local_projection[group])
                bad_accumulation += \
                    (10 * time_complexity) + \
                    (1/(100 * neighbour_distances[group])) + \
                    (100 * intrusion_damages[group]) + \
                    (10 * surrounded_damages[group]) + \
                    (1000 * collision_damages[group]) 
            fitness_value = bad_accumulation/len(self.classes)
        except:
            return self._bad_fitness()
        # return fitness value
        fitness = {}
        fitness['default'] = fitness_value
        return fitness
    
    def get_new_features(self):
        best_phenotypes = self.best_individuals(len(self.classes), benchmark='default', representation='phenotype')
        new_features = []
        for best_phenotype in best_phenotypes:
            if not (best_phenotype in new_features):
                new_features.append(best_phenotype)
        return new_features
    
    def get_svm(self):
        svc = None
        try:
            svc = svm.SVC(max_iter=2000, class_weight='auto')
        except:
            svc = svm.SVC()
        return svc 


class GE_Multi_Fitness_GA_SVM(GE_Multi_Fitness):
    def __init__(self):
        super(GE_Multi_Fitness_GA_SVM, self).__init__()
        self.ga_svm = GA_SVM()
        
    def process(self):
        GE_Multi_Fitness.process(self)
        
        # define new features
        new_features = []
        for group in self.classes:
            best_phenotype = self.best_individuals(1, benchmark=group, representation='phenotype')
            if not (best_phenotype in new_features):
                new_features.append(best_phenotype)
        # start GA_SVM
        self.ga_svm.label = 'GA SVM Part'
        self.ga_svm.classes = self.classes
        self.ga_svm.variables = new_features
        self.ga_svm.training_data = build_new_data(self.training_data, self.variables, new_features)
        self.ga_svm.training_num_target = self.get_num_targets()
        self.ga_svm.process()
    
    def get_new_features(self):
        return self.ga_svm.get_new_features()
    
    def get_svm(self):
        return self.ga_svm.get_svm()

class GE_Global_Fitness_GA_SVM(GE_Global_Fitness):
    def __init__(self):
        super(GE_Global_Fitness_GA_SVM, self).__init__()
        self.ga_svm = GA_SVM()
        
    def process(self):
        GE_Global_Fitness.process(self)
        
        # define new features
        new_features = []
        best_phenotypes = self.best_individuals(len(self.classes), benchmark='default', representation='phenotype')
        for best_phenotype in best_phenotypes:
            if not (best_phenotype in new_features):
                new_features.append(best_phenotype)
        # start GA_SVM
        self.ga_svm.label = 'GA SVM Part'
        self.ga_svm.classes = self.classes
        self.ga_svm.variables = new_features
        self.ga_svm.training_data = build_new_data(self.training_data, self.variables, new_features)
        self.ga_svm.training_num_target = self.get_num_targets()
        self.ga_svm.process()
    
    def get_new_features(self):
        return self.ga_svm.get_new_features()
    
    def get_svm(self):
        return self.ga_svm.get_svm()
    
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
        labels.sort()
        for label in label_targets:
            training_count_per_fold[label] = math.floor(len(target_indexes[label])/self.fold)
        
        # folding scenario
        output = ''
        
        all_svm_results = {
            'original_svm' : [],            
            'ge_global_fitness' : [],
            'ge_multi_fitness' : [],
            'ga_svm' : []
        }
        
        try:
            os.mkdir(self.label)
        except:
            shutil.rmtree(self.label)
            os.mkdir(self.label)
        
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
            
            if self.fold == 1:
                test_data = training_data
                test_label_targets = training_label_targets
                test_num_targets = training_num_targets
            
            print('FOLD : '+str(fold_index+1))
            
            
            # Original SVM
            original_svm_label = 'Original SVM "'+self.label+'" Fold '+str(fold_index+1)
            svm_result = get_svm_result(training_data, training_num_targets, test_data, test_num_targets, variables, variables, original_svm_label)
            output += svm_result['str']
            all_svm_results['original_svm'].append(svm_result)
            draw_projection(training_data, training_label_targets, variables, variables, original_svm_label+' Training', self.label+'/'+original_svm_label+' Training Feature Projection.png')
            draw_projection(test_data, test_label_targets, variables, variables, original_svm_label+' Test', self.label+'/'+original_svm_label+' Test Feature Projection.png')
            
            # GA SVM
            ga_svm = GA_SVM()
            ga_svm.classes = labels
            ga_svm.variables = variables
            ga_svm.training_data = training_data
            ga_svm.training_num_target = training_num_targets
            ga_svm.label = 'GA SVM "'+self.label+'" Fold '+str(fold_index+1)
            ga_svm.stopping_value = 0
            ga_svm.max_epoch = self.max_epoch
            ga_svm.population_size = self.population_size
            ga_svm.process()
            new_features = ga_svm.get_new_features()
            svc = ga_svm.get_svm()
            svm_result = get_svm_result(training_data, training_num_targets, test_data, test_num_targets, variables, new_features, ga_svm.label, svc)
            output += svm_result['str']
            all_svm_results['ga_svm'].append(svm_result)
            ga_svm.show(True, self.label+'/'+ga_svm.label+'.png')
            draw_projection(training_data, training_label_targets, variables, new_features, ga_svm.label+' Training', self.label+'/'+ga_svm.label+' Training Feature Projection.png')
            draw_projection(test_data, test_label_targets, variables, new_features, ga_svm.label+' Test', self.label+'/'+ga_svm.label+' Test Feature Projection.png')
            
            # GE Global-Fitness (My Previous Research)
            ge_global_fitness = GE_Global_Fitness()
            ge_global_fitness.classes = labels
            ge_global_fitness.variables = variables
            ge_global_fitness.training_data = training_data
            ge_global_fitness.training_target = training_label_targets
            ge_global_fitness.label = 'GE Global Fitness "'+self.label+'" Fold '+str(fold_index+1)
            ge_global_fitness.stopping_value = 1.0
            ge_global_fitness.max_epoch = self.max_epoch
            ge_global_fitness.individual_length = 30
            ge_global_fitness.population_size = self.population_size
            ge_global_fitness.process()
            # new features
            new_features = ge_global_fitness.get_new_features()
            svc = ge_global_fitness.get_svm()
            svm_result = get_svm_result(training_data, training_num_targets, test_data, test_num_targets, variables, new_features, ge_global_fitness.label, svc)
            output += svm_result['str']
            all_svm_results['ge_global_fitness'].append(svm_result)
            ge_global_fitness.show(True, self.label+'/'+ge_global_fitness.label+'.png')
            draw_projection(training_data, training_label_targets, variables, new_features, ge_global_fitness.label+' Training', self.label+'/'+ge_global_fitness.label+' Training Feature Projection.png')
            draw_projection(test_data, test_label_targets, variables, new_features, ge_global_fitness.label+' Test', self.label+'/'+ge_global_fitness.label+' Test Feature Projection.png')
            
            # GE Multi-Fitness (My Fallen Hero)
            ge_multi_fitness = GE_Multi_Fitness()
            ge_multi_fitness.classes = labels
            ge_multi_fitness.variables = variables
            ge_multi_fitness.training_data = training_data
            ge_multi_fitness.training_target = training_label_targets
            ge_multi_fitness.label = 'GE Multi Fitness "'+self.label+'" Fold '+str(fold_index+1)
            ge_multi_fitness.stopping_value = 1.0
            ge_multi_fitness.max_epoch = self.max_epoch
            ge_multi_fitness.individual_length = 30
            ge_multi_fitness.population_size = self.population_size
            ge_multi_fitness.process()
            # new features
            new_features = ge_multi_fitness.get_new_features()
            svc = ge_multi_fitness.get_svm()
            svm_result = get_svm_result(training_data, training_num_targets, test_data, test_num_targets, variables, new_features, ge_multi_fitness.label, svc)
            output += svm_result['str']
            all_svm_results['ge_multi_fitness'].append(svm_result)
            ge_multi_fitness.show(True, self.label+'/'+ge_multi_fitness.label+'.png')
            draw_projection(training_data, training_label_targets, variables, new_features, ge_multi_fitness.label+' Training', self.label+'/'+ge_multi_fitness.label+' Training Feature Projection.png')
            draw_projection(test_data, test_label_targets, variables, new_features, ge_multi_fitness.label+' Test', self.label+'/'+ge_multi_fitness.label+' Test Feature Projection.png')
            
            del ga_svm
            del ge_global_fitness
            del ge_multi_fitness
            del training_data           
            del training_label_targets
            del training_num_targets
            del test_data
            del test_label_targets
            del test_num_targets
            del training_indexes
            gc.collect()
            
        print output
        
        text_file = open(self.label+'/SVM training and test comparison '+self.label+'.txt', "w")
        text_file.write(output)
        text_file.close()
        
        fig = plt.figure(figsize=(20.0, 12.0))
        sp_1 = fig.add_subplot(1, 2, 1)
        sp_2 = fig.add_subplot(1, 2, 2)
        
        training_accuracy = {}
        test_accuracy = {}
        for label in all_svm_results:
            training_accuracy[label] = []
            test_accuracy[label] = []
        
        for i in xrange(self.fold):
            sp_1.plot([i,i], [0, 100], 'k--')
            sp_2.plot([i,i], [0, 100], 'k--')            
            for label in all_svm_results:
                training_accuracy[label].append(all_svm_results[label][i]['training_result']['accuracy'])
                test_accuracy[label].append(all_svm_results[label][i]['test_result']['accuracy'])
            
        fold_indexes = list(xrange(self.fold))
        for label in all_svm_results:
            if len(fold_indexes)==1:
                sp_1.plot(fold_indexes, training_accuracy[label], label=label, marker='o')
                sp_2.plot(fold_indexes, test_accuracy[label], label=label, marker='o')
            else:
                sp_1.plot(fold_indexes, training_accuracy[label], label=label)
                sp_2.plot(fold_indexes, test_accuracy[label], label=label)
        
        x_range = self.fold
        
        sp_1.set_title('Training Accuration')
        sp_1.set_ylabel('Accuration (%)')
        sp_1.set_xlabel('Fold')
        sp_1.set_ylim(-1,101)
        sp_1.set_xlim(0-0.1*x_range, (self.fold-1)+0.1*x_range)
        sp_1.legend(shadow=True, loc=0)
        
        sp_2.set_title('Test Accuration')
        sp_2.set_ylabel('Accuration (%)')
        sp_2.set_xlabel('Fold')
        sp_2.set_ylim(-1,101)
        sp_2.set_xlim(0-0.1*x_range, (self.fold-1)+0.1*x_range)
        sp_2.legend(shadow=True, loc=0)
        
        x_formatter = Draw_accuracy_x_formatter(self.fold)
        sp_1.xaxis.set_major_formatter(x_formatter)
        sp_2.xaxis.set_major_formatter(x_formatter)
        
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        plt.suptitle('SVM training and test comparison of '+self.label)

        plt.savefig(self.label+'/'+'SVM training and test comparison "'+self.label+'".png', dpi=100)
        
        fig.clf()
        plt.close()
        gc.collect()

def feature_extracting(data, features, label='data', max_epoch=200, population_size=100, fold=5):
    fe = Feature_Extractor()
    fe.label = label
    fe.max_epoch = max_epoch
    fe.records = data
    fe.population_size = population_size
    fe.fold = fold
    fe.variables = features
    fe.measurement = 'error'
    fe.process()
    return fe

def extract_csv(csv_file_name, delimiter=','):
    r = csv.reader(open(csv_file_name), delimiter=delimiter)
    r = list(r)
    variables = r[0]
    variables = variables[:-1]
    data = r[1:]
    return {'variables':variables,'data':data}