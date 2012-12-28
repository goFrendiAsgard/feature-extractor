import os, sys, gc, shutil
lib_path = os.path.abspath('./gogenpy')
sys.path.insert(0,lib_path)

import csv
import math, numpy
from gogenpy import classes, utils
from sklearn import svm
import matplotlib.pyplot as plt

LIMIT_ZERO = utils.LIMIT_ZERO

def extract_csv(csv_file_name, delimiter=','):
    r = csv.reader(open(csv_file_name), delimiter=delimiter)
    r = list(r)
    return r

def shuffle_record(record):
    record_length = len(record)
    i = 0
    while i<record_length:
        rnd1 = utils.randomizer.randrange(1,record_length)
        rnd2 = utils.randomizer.randrange(1,record_length)
        record[rnd1], record[rnd2] = record[rnd2], record[rnd1]
        i+=1
    return record

class Feature_Extractor(object):   
    def __init__(self, records, fold_count=1, fold_index=0):
        self.label = ''
        self.features = []
        self.data = []
        self.label_target = [] 
        self.num_target = []
        self.group_label = []
        self.group_count = {}        
        self.fold_count = fold_count
        self.fold_index = fold_index
        self.training_data = []
        self.training_label_target = []
        self.training_num_target = []
        self.test_data = []
        self.test_label_target = []
        self.test_num_target = []
        
        # preprocess everything
        self.data = []
        self.label_target = []
        self.num_target = []
        self.group_label = []
        self.group_count = {}
        target_dictionary = {}
        group_index = 0
        for i in xrange(len(records)):
            record = records[i]
            if i==0:
                self.features = record[:-1]
            else:
                self.data.append(record[:-1])
                self.label_target.append(record[-1])
                if not (record[-1] in target_dictionary):
                    target_dictionary[record[-1]] = group_index
                    self.group_count[record[-1]] = 0
                    self.group_label.append(record[-1])              
                    group_index += 1
                self.group_count[record[-1]] += 1           
                self.num_target.append(target_dictionary[record[-1]])                
        self.group_label.sort()
        
        # prepare fold
        self.training_data = []
        self.training_label_target = []
        self.training_num_target = []
        self.test_data = []
        self.test_label_target = []
        self.test_num_target = []
        if self.fold_count == 1:
            self.training_data = self.data
            self.training_label_target = self.label_target
            self.training_num_target = self.num_target
            self.test_data = self.data
            self.test_label_target = self.label_target
            self.test_num_target = self.num_target
        else:
            group_indexes = {}
            for i in xrange(len(self.data)):
                group = self.label_target[i]
                if not group in group_indexes:
                    group_indexes[group] = []
                group_indexes[group].append(i)
            for group in self.group_label:
                data_count_per_fold = math.ceil(len(group_indexes[group])/self.fold_count)
                lower_bound = int(self.fold_index * data_count_per_fold)
                upper_bound = int((self.fold_index+1) * data_count_per_fold)
                test_indexes = group_indexes[group][lower_bound:upper_bound]
                for i in group_indexes[group]:
                    if i in test_indexes:
                        self.test_data.append(self.data[i])
                        self.test_label_target.append(self.label_target[i])
                        self.test_num_target.append(self.num_target[i])
                    else:
                        self.training_data.append(self.data[i])
                        self.training_label_target.append(self.label_target[i])
                        self.training_num_target.append(self.num_target[i])
    
    def _get_projection(self, new_feature, used_data = None):
        all_result = []
        all_error = []
        for data in self.data:
            result, error = utils.execute(new_feature, data, self.features)            
            all_error.append(error)
            if error:
                all_result.append(None)
            else:
                all_result.append(result)
        all_is_none = True
        for i in all_result:
            if i is not None:
                all_is_none = False
                break
        if all_is_none:
            min_result = 0
            max_result = 1
        else:
            min_result = min(x for x in all_result if x is not None)
            max_result = max(x for x in all_result if x is not None)
        result_range = max(max_result-min_result, LIMIT_ZERO)
        if used_data is None: # include all data
            for i in xrange(len(all_result)):
                if all_error[i]:
                    all_result[i] = -1
                else:
                    all_result[i] = (all_result[i]-min_result)/result_range
            return all_result
        else:
            used_result = []
            for i in xrange(len(used_data)):
                result, error = utils.execute(new_feature, used_data[i], self.features)
                if error:
                    used_result.append(-1)
                else:
                    used_result.append((result-min_result)/result_range)
            return used_result
        
    def _calculate_histogram(self, data):
        histogram = {}
        for value in data:
            if not value in histogram:
                histogram[value] = 0.0
            histogram[value]+=1
        return histogram
        
class Genetics_Feature_Extractor(Feature_Extractor, classes.GA_Base):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        Feature_Extractor.__init__(self, records, fold_count, fold_index)
        classes.GA_Base.__init__(self)
        if classifier is None:
            try:
                self.classifier = svm.SVC(max_iter=2000, class_weight='auto')
            except:
                self.classifier = svm.SVC(class_weight='auto')
        else:
            self.classifier = classifier
        
    def get_new_features(self):
        return self.features
    
    def get_metrics(self, new_feature):
        target = self.label_target
        projection = self._get_projection(new_feature, self.training_data)
        hist_group = {}
        mean_group = {}
        stdev_group = {}
        projection_group_count = {}
        separability_index = {}
        total_hist = self._calculate_histogram(projection)
        for group in self.group_label:
            group_projection = []
            for i in xrange(len(projection)):
                if group == target[i]:
                    group_projection.append(projection[i])
            projection_group_count[group] = len(group_projection)
            hist_group[group] = self._calculate_histogram(group_projection)
            mean_group[group] = numpy.mean(group_projection)
            stdev_group [group]= numpy.std(group_projection)
        
        # separability index (0 for not separable, 1 for greatly separable)
        for current_group in self.group_label:
            current_count = projection_group_count[current_group]
            current_separability_index = 0.0
            count = 0.0
            good_neighbour_count = 0.0
            
            for current_value in hist_group[current_group]:
                # get distance rank to current value
                distances = []
                for all_value in total_hist:
                    distance = abs(current_value - all_value)
                    distances.append(distance)
                distances.sort()
                
                # calculate good neighbour_count and total count                
                for i in xrange(len(distances)):
                    if count >= current_count:
                        break
                    distance = distances[i]
                    # + distance
                    value = current_value + distance
                    if value in total_hist:
                        count += total_hist[value]
                        if value in hist_group[current_group]:
                            good_neighbour_count += hist_group[current_group][value]
                    # - distance
                    value = current_value - distance
                    if value in total_hist:
                        count += total_hist[value]
                        if value in hist_group[current_group]:
                            good_neighbour_count += hist_group[current_group][value]
            # current_separability
            current_separability_index += good_neighbour_count/max(count,LIMIT_ZERO)
            separability_index[current_group] = current_separability_index
            
        # get metric
        metric = {
            'group_histogram':hist_group,
            'total_histogram':total_hist,
            'separability_index':separability_index,
            'mean':mean_group,
            'stdev':stdev_group
        }
        return metric                    
                
        
    
    def get_accuracy(self, new_features=None):
        if (new_features is None):
            new_features = self.get_new_features()
        new_feature_count = len(new_features)
        if new_feature_count == 0:
            return {'training':0, 'test':0, 'total':0}
        
        training_data = self.training_data
        training_target = self.training_num_target
        test_data = self.test_data
        test_target = self.test_num_target
        new_training_data = []
        new_test_data = []
        for i in xrange(len(training_data)):
            new_training_data.append([0]*new_feature_count)
        for i in xrange(len(test_data)):
            new_test_data.append([0]*new_feature_count)
        for i in xrange(new_feature_count):
            feature = new_features[i]
            training_projection = self._get_projection(feature,training_data)
            test_projection = self._get_projection(feature,test_data)
            for j in xrange(len(training_data)):
                new_training_data[j][i] = training_projection[j]
            for j in xrange(len(test_data)):
                new_test_data[j][i] = test_projection[j]
        self.classifier.fit(new_training_data,training_target)
        training_prediction = self.classifier.predict(new_training_data)
        test_prediction = self.classifier.predict(new_test_data)
        training_mistake = 0.0
        test_mistake = 0.0
        for i in xrange(len(training_target)):
            if int(training_target[i]) <> int(training_prediction[i]):
                training_mistake+=1
        for i in xrange(len(test_target)):
            if int(test_target[i]) <> int(test_prediction[i]):
                test_mistake+=1
        training_accuracy = (len(training_target)-training_mistake)/len(training_target)
        test_accuracy = (len(test_target)-test_mistake)/len(test_target)
        total_accuracy = ((len(training_target)+len(test_target))-(training_mistake+test_mistake))/(len(training_target)+len(test_target))
        return {'training':training_accuracy, 'test':test_accuracy, 'total':total_accuracy}

class GA_Select_Feature(Genetics_Feature_Extractor, classes.Genetics_Algorithm):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        Genetics_Feature_Extractor.__init__(self, records, fold_count=1, fold_index=0, classifier=None)     
        classes.Genetics_Algorithm.__init__(self)
        self.fitness_measurement = 'MAX'
        self.representations = ['default']
        self.benchmarks = ['accuration']
        self.individual_length = len(self.features)
    
    def gene_to_features(self, gene):
        new_features = []
        for i in xrange(len(self.features)):
            if gene[i] == '1':
                feature = self.features[i]
                new_features.append(feature)
        return new_features
        
    def do_calculate_fitness(self, individual):
        new_features = self.gene_to_features(individual['default'])
        accuration = self.get_accuracy(new_features)
        return {'accuration':accuration['training']}
    
    def get_new_features(self):
        best_gene = self.best_individuals(1, 'accuration', 'default')
        return self.gene_to_features(best_gene)

class GP_Select_Feature(Genetics_Feature_Extractor, classes.Genetics_Programming):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        Genetics_Feature_Extractor.__init__(self, records, fold_count=1, fold_index=0, classifier=None)     
        classes.Genetics_Programming.__init__(self)
        self.fitness_measurement = 'MAX'
        self.benchmarks = ['accuration']
        self.nodes = [
            self.features,
            ['exp','sigmoid','abs','sin','cos','sqr','sqrt'],
            ['plus','minus','multiply','divide']
        ]
    
    def do_calculate_fitness(self, individual):
        new_feature = individual['phenotype']
        accuration = self.get_accuracy([new_feature])
        return {'accuration':accuration['training']}
    
    def get_new_features(self):
        best_phenotype = self.best_individuals(1, 'accuration', 'phenotype')
        return [best_phenotype]

class GE_Select_Feature(Genetics_Feature_Extractor, classes.Grammatical_Evolution):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        Genetics_Feature_Extractor.__init__(self, records, fold_count=1, fold_index=0, classifier=None)     
        classes.Grammatical_Evolution.__init__(self)
        self.fitness_measurement = 'MAX'
        self.benchmarks = ['accuration']
        self.variables = self.features
        self.grammar = {
            '<expr>' : ['<var>','<expr> <op> <expr>','<func>(<expr>)'],
            '<var>'  : self.variables,
            '<op>'   : ['+','-','*','/'],
            '<func>' : ['exp','sigmoid','abs','sin','cos','sqr','sqrt']
        }
        self.start_node = '<expr>'
    
    def do_calculate_fitness(self, individual):
        new_feature = individual['phenotype']
        accuration = self.get_accuracy([new_feature])
        return {'accuration':accuration['training']}
    
    def get_new_features(self):
        best_phenotype = self.best_individuals(1, 'accuration', 'phenotype')
        return [best_phenotype]

class Global_Separability_Fitness(Genetics_Feature_Extractor):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        self.benchmarks = ['separability']
    
    def get_new_features(self):
        best_phenotype = self.best_individuals(1, 'separability', 'phenotype')
        return [best_phenotype]
    
    def do_calculate_fitness(self, individual):
        feature = individual['phenotype']
        metrics = self.get_metrics(feature)
        separability_index = metrics['separability_index']
        stdev = metrics['stdev']
        mean = metrics['mean']
        average_minimum_mean_distance = 0.0
        average_stdev = 0.0
        average_separability_index = 0.0
        for current_group in self.group_label:
            minimum_mean_distance = 10
            for compare_group in self.group_label:
                if current_group == compare_group:
                    continue
                mean_distance = abs(mean[current_group]-mean[compare_group])
                if mean_distance<minimum_mean_distance:
                    minimum_mean_distance = mean_distance
            average_minimum_mean_distance += minimum_mean_distance
            average_stdev += stdev[current_group]
            average_separability_index += separability_index[current_group]
        average_minimum_mean_distance /= len(self.group_label)
        average_stdev /= len(self.group_label)
        average_separability_index /= len(self.group_label)
        
        # stdev cannot surpass range. Since the range has been normalized between 0-1 (and -1) for error,
        # I think it is save to use 2.0-average_stdev
        fitness = {}
        fitness['separability'] =\
            5 * (average_minimum_mean_distance) +\
            10 * (average_separability_index) +\
            (2.0-average_stdev)
        return fitness

class Local_Separability_Fitness(Genetics_Feature_Extractor):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        self.benchmarks = self.group_label
    
    def get_new_features(self):
        new_features = []
        for benchmark in self.benchmarks:
            best_phenotype = self.best_individuals(1, benchmark, 'phenotype')
            new_features.append(best_phenotype)
        return new_features
    
    def do_calculate_fitness(self, individual):
        feature = individual['phenotype']
        metrics = self.get_metrics(feature)
        separability_index = metrics['separability_index']
        stdev = metrics['stdev']
        mean = metrics['mean']
        fitness = {}
        for current_group in self.group_label:
            # minimum mean
            local_minimum_mean_distance = 10
            for compare_group in self.group_label:
                if current_group == compare_group:
                    continue
                mean_distance = abs(mean[current_group]-mean[compare_group])
                if mean_distance<local_minimum_mean_distance:
                    local_minimum_mean_distance = mean_distance
            # separability
            local_separability_index = separability_index[current_group]
            local_stdev = stdev[current_group]
            local_fitness =\
                5 * (local_minimum_mean_distance) +\
                10 * (local_separability_index) +\
                (2.0-local_stdev)
            fitness[current_group] = local_fitness
        return fitness

class GP_Global_Separability_Fitness(GP_Select_Feature, Global_Separability_Fitness):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GP_Select_Feature.__init__(self, records, fold_count=1, fold_index=0, classifier=None)
        Global_Separability_Fitness.__init__(self, records, fold_count, fold_index, classifier)
    
    def do_calculate_fitness(self, individual):
        return Global_Separability_Fitness.do_calculate_fitness(self, individual)
    
    def get_new_features(self):
        return Global_Separability_Fitness.get_new_features(self)

class GP_Local_Separability_Fitness(GP_Select_Feature, Local_Separability_Fitness):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GP_Select_Feature.__init__(self, records, fold_count=1, fold_index=0, classifier=None)
        Local_Separability_Fitness.__init__(self, records, fold_count, fold_index, classifier)
    
    def do_calculate_fitness(self, individual):
        return Local_Separability_Fitness.do_calculate_fitness(self, individual)
    
    def get_new_features(self):
        return Local_Separability_Fitness.get_new_features(self)

class GE_Global_Separability_Fitness(GE_Select_Feature, Global_Separability_Fitness):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GE_Select_Feature.__init__(self, records, fold_count=1, fold_index=0, classifier=None)
        Global_Separability_Fitness.__init__(self, records, fold_count, fold_index, classifier)
    
    def do_calculate_fitness(self, individual):
        return Global_Separability_Fitness.do_calculate_fitness(self, individual)
    
    def get_new_features(self):
        return Global_Separability_Fitness.get_new_features(self)

class GE_Local_Separability_Fitness(GE_Select_Feature, Local_Separability_Fitness):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GE_Select_Feature.__init__(self, records, fold_count=1, fold_index=0, classifier=None)
        Local_Separability_Fitness.__init__(self, records, fold_count, fold_index, classifier)
    
    def do_calculate_fitness(self, individual):
        return Local_Separability_Fitness.do_calculate_fitness(self, individual)
    
    def get_new_features(self):
        return Local_Separability_Fitness.get_new_features(self)
        
def extract_feature(records, data_label='Test', fold_count=5, extractors=[], classifier=None):
    if len(extractors) == 0:
        extractors = [
            {'class': GA_Select_Feature, 'label':'GA', 'color':'red'},
            {'class': GP_Select_Feature, 'label':'GP', 'color':'orange'},
            {'class': GP_Global_Separability_Fitness, 'label':'GP Global', 'color':'green'},
            {'class': GP_Local_Separability_Fitness, 'label':'GP Local', 'color':'blue'},
            {'class': GE_Select_Feature, 'label':'GE', 'color':'cyan'},
            {'class': GE_Global_Separability_Fitness, 'label':'GE Global', 'color':'magenta'},
            {'class': GE_Local_Separability_Fitness, 'label':'GE Local', 'color':'black'}
        ]
        
    # prepare directory
    try:
        os.mkdir(data_label)
    except:
        shutil.rmtree(data_label)
        os.mkdir(data_label)
    
    all_accuracy = {}
    accuracy_metrics = ['training','test','total']
    minimum_accuracy = 100
    maximum_accuracy = 0
        
    # process every fold with every extractor
    for fold_index in xrange(fold_count):
        for extractor in extractors:
            
            output = ''
            extractor_class = extractor['class']
            extractor_label = extractor['label']
            fe = extractor_class(records, fold_count, fold_index, classifier)
            fe.label = extractor_label+' (Fold '+str(fold_index+1)+' of '+str(fold_count)+')'
            fe.process()            
            new_features = fe.get_new_features()
            accuracy = fe.get_accuracy()
            
            # prepare output
            groups = fe.group_label            
            max_group_label_length = 0
            for group in groups:
                if len(group)>max_group_label_length:
                    max_group_label_length = len(group)
            
            # prepare all_accuracy for accuracy plotting
            if fold_index == 0:
                all_accuracy[extractor_label] = {'training':[],'test':[],'total':[]}
            for label in accuracy_metrics:
                accuracy_value = accuracy[label]
                all_accuracy[extractor_label][label].append(accuracy_value)
                if accuracy_value<minimum_accuracy:
                    minimum_accuracy = accuracy_value
                if accuracy_value>maximum_accuracy:
                    maximum_accuracy = accuracy_value
            
            
            # write output
            output += 'Accuracy :\r\n'
            output += '\r\n'
            for label in accuracy_metrics:
                accuracy_value = accuracy[label]
                while len(label)<8:
                    label += ' '
                output += '  '+label+ ' : '+str(accuracy_value)+'\r\n'
            output += '\r\n\r\n'
            output += str(len(new_features))+' Feature(s) Used :\r\n'
            output += '\r\n'
            total_histogram = {}
            group_histogram = {}
            for feature in new_features:
                output += '  '+feature+'\r\n'
                metrics = fe.get_metrics(feature)
                separability_index = metrics['separability_index']
                stdev = metrics['stdev']
                mean = metrics['mean']
                total_histogram[feature] = metrics['total_histogram']
                group_histogram[feature] = metrics['group_histogram']
                output += '    Separability Index Per Class :\r\n'
                for group in groups:
                    label_group = group
                    while len(label_group)<max_group_label_length:
                        label_group += ' '
                    output += '      '+label_group+' : '+str(separability_index[group])+'\r\n'
                output += '    Standard Deviation Per Class :\r\n'
                for group in groups:
                    label_group = group
                    while len(label_group)<max_group_label_length:
                        label_group += ' '
                    output += '      '+label_group+' : '+str(stdev[group])+'\r\n'
                output += '    Mean Per Class :\r\n'
                for group in groups:
                    label_group = group
                    while len(label_group)<max_group_label_length:
                        label_group += ' '
                    output += '      '+label_group+' : '+str(mean[group])+'\r\n'
                output += '\r\n'
            print output
            fe.show(True,data_label+'/'+extractor_label+' Fold '+str(fold_index+1)+'.png')
            text_file = open(data_label+'/'+extractor_label+' Fold '+str(fold_index+1)+'.txt', "w")
            text_file.write(output)
            text_file.close()
            
            # plot features
            feature_count = len(new_features)
            if feature_count == 1:
                col = 1
            elif feature_count <=6:
                col = 2
            else:
                col = 3
            if math.fmod(feature_count,col)>0:
                row = int(feature_count/col)+1
            else:
                row = int(feature_count/col)
            fig = plt.figure(figsize=(25.0, 12.0))
            plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
            plt.suptitle('Feature Projection')          
            for i in xrange(feature_count):
                sp = fig.add_subplot(row,col,i+1)
                feature = new_features[i]
                feature_total_histogram = total_histogram[feature]
                feature_local_histogram = group_histogram[feature]
                # get max count of total histogram
                total_max_count = 0
                total_max_value = -1
                total_min_value = 1
                for value in feature_total_histogram:
                    if feature_total_histogram[value]>total_max_count:
                        total_max_count = feature_total_histogram[value]
                    if value>total_max_value:
                        total_max_value = value
                    if value<total_min_value:
                        total_min_value = value
                
                
                group_count = len(feature_local_histogram)
                
                # draw blue lines
                group_index = 1
                for group in groups:
                    local_group_histogram = feature_local_histogram[group]
                    group_max_value = -1
                    group_min_value = 1
                    for value in local_group_histogram:
                        if value>group_max_value:
                            group_max_value = value
                        if value<group_min_value:
                            group_min_value = value
                    sp.plot([group_min_value, group_min_value],[0, group_index],'b--')
                    sp.plot([group_max_value, group_max_value],[0, group_index],'b--')
                    group_index += 1 
                
                # draw local histograms
                group_index = 1
                for group in groups:
                    local_group_histogram = feature_local_histogram[group]
                    group_max_value = -1
                    group_min_value = 1
                    for value in local_group_histogram:
                        count = local_group_histogram[value]
                        if value>group_max_value:
                            group_max_value = value
                        if value<group_min_value:
                            group_min_value = value
                        sp.plot([value,value],[group_index, group_index+0.9*count/total_max_count],'k', linewidth=2)
                    sp.plot([group_min_value, group_max_value],[group_index, group_index],'k')
                    group_index += 1                      
                
                
                # plot global histogram
                sp.plot([total_min_value,total_max_value],[0,0],'k')
                for value in feature_total_histogram:
                    count = feature_total_histogram[value]
                    sp.plot([value,value], [0, 0.9*count/total_max_count],'k', linewidth=2)                
                
                    
                sp.set_ylim(-0.1,group_count+1+0.1)
                sp.set_xlim(total_min_value-0.1, total_max_value+0.1)
                feature_title = feature
                if len(feature)>40:
                    feature_title = feature[:40]+' ...'                
                sp.set_title(feature_title)
                # set yticks
                pos = list(range(group_count+1))
                group_names = list(['global']+groups)
                plt.yticks(pos,group_names, rotation=45)
            
    
            plt.savefig(data_label+'/'+extractor_label+' Feature Projection Fold '+str(fold_index+1)+'".png', dpi=100)
            fig.clf()
            plt.close()
            gc.collect()
    
    # accuracy plotting
    minimum_accuracy-=0.01
    maximum_accuracy+= 0.01
    extractor_count = len(extractors)
    metric_count = len(accuracy_metrics)
    min_x = -0.5
    max_x = (extractor_count)*fold_count-0.5
    extractor_labels = []    
    for i in xrange(extractor_count):
        extractor_labels.append(extractors[i]['label'])
     
    
    fig = plt.figure(figsize=(20.0, 15.0))
    plt.subplots_adjust(hspace = 0.15, wspace = 0.5)
    plt.suptitle('Accuracy')
    
    for i in xrange(metric_count):
        metric_label = accuracy_metrics[i]
        sp = fig.add_subplot(3,1,i+1)
        # fold margin line
        for fold_index in xrange(fold_count):
            if fold_index == 0:
                continue
            x = (fold_index)*extractor_count-0.5
            sp.plot([x,x],[minimum_accuracy*100,maximum_accuracy*100],'k')
        # bar
        for extractor_index in xrange(extractor_count):
            for fold_index in xrange(fold_count):
                color = extractors[extractor_index]['color']
                label = extractors[extractor_index]['label']
                value = all_accuracy[label][metric_label][fold_index]
                x = fold_index * extractor_count + extractor_index
                sp.plot([x,x],[minimum_accuracy*100,value*100],color=color, linewidth=3)
        # helper line
        for extractor_index in xrange(extractor_count):
            for fold_index in xrange(fold_count):
                value = all_accuracy[label][metric_label][fold_index]
                sp.plot([min_x, max_x],[value*100,value*100],'k--')
                
        sp.set_ylim(minimum_accuracy*100,maximum_accuracy*100)
        sp.set_xlim(min_x, max_x)
        sp.set_title(metric_label)
        if i<metric_count-1:
            pos = list(range(extractor_count*fold_count))
            extractor_names = list(['']*extractor_count*fold_count)
            plt.xticks(pos,extractor_names, rotation=45)
        else:
            pos = list(range(extractor_count*fold_count))
            extractor_names = list(extractor_labels*fold_count)
            plt.xticks(pos,extractor_names, rotation=45)
        
        
    plt.savefig(data_label+'/Accuracy.png', dpi=100)
    fig.clf()
    plt.close()
    gc.collect()