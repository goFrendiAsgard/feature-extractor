import os, sys, gc, shutil
lib_path = os.path.abspath('./gogenpy')
sys.path.insert(0,lib_path)

import csv
import math, numpy
from gogenpy import utils
from gogenpy import classes
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pickle

LIMIT_ZERO = utils.LIMIT_ZERO

def extract_csv(csv_file_name, delimiter=','):
    '''
    Extract csv file into records.
    
    Usage:
    >>> records = extract_csv('iris-data.csv')
    >>> print records
    [['petal_width', 'petal_length', 'sepal_width', 'sepal_length', 'class'],
     [5.1,3.5,1.4,0.2,'Iris-setosa'],
     ...
    ]
    '''
    r = csv.reader(open(csv_file_name), delimiter=delimiter)
    r = list(r)
    return r

def shuffle_record(record):
    '''
    Shuffle records, assuming the first row is header and therefore untouched
    
    Usage:
    >>> records = extract_csv('iris-data.csv')
    >>> records = shuffle_record(records)
    '''
    record_length = len(record)
    i = 0
    while i<record_length:
        rnd1 = utils.randomizer.randrange(1,record_length)
        rnd2 = utils.randomizer.randrange(1,record_length)
        record[rnd1], record[rnd2] = record[rnd2], record[rnd1]
        i+=1
    return record

def get_projection(new_feature, old_features, all_data, used_data = None, used_target = None):
    '''
    Get projection of used_data (with old_features) in new_feature.
    If used_data is None, then all-data will be used as used_data
    If used_target is None, a list of projection will be returned. 
    if used_target is not None, then a dictionary with every class in target as key, 
    and projection of corresponding class as value will be returned
    
    Usage:
    >>> new_feature = 'x+y'
    >>> old_features = [x,y]
    >>> all_data = [[1,2],[1,3],[2,4],[1,5]]
    >>>
    >>> # without used_data and used_target
    >>> projection = get_projection(new_feature, old_features, all_data)
    >>> print projection
    [3,4,6,6]
    >>>
    >>> # with used_data, without used_target
    >>> used_data = [[1,2],[1,5]]
    >>> projection = get_projection(new_feature, old_features, all_data, used_data)
    >>> print projection
    [3,6]
    >>>
    >>> # with used_target
    >>> target = ['small','small','big','big']
    >>> projection = get_projection(new_feature, old_features, all_data, used_target=target)
    >>> print projection
    {'small':[3,4],'big':[6,6]}
    '''
    used_projection = []
    all_result = []
    all_error = []
    # get all result
    for data in all_data:
        result, error = utils.execute(new_feature, data, old_features)
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
    if max_result-min_result>0:
        result_range = max_result-min_result
    else:
        result_range = LIMIT_ZERO
    if used_data is None: # include all data
        for i in xrange(len(all_result)):
            if all_error[i]:
                all_result[i] = -1
            else:
                all_result[i] = (all_result[i]-min_result)/result_range
        used_projection = all_result
    else:
        used_result = []
        for i in xrange(len(used_data)):
            result, error = utils.execute(new_feature, used_data[i], old_features)
            if error:
                used_result.append(-1)
            else:
                used_result.append((result-min_result)/result_range)
        used_projection = used_result
    
    # make is safer, only allow int and float to be the member of used_projection
    for i in xrange(len(used_projection)):
        value = used_projection[i]
        if (not isinstance(value,float)) and (not isinstance(value,int)):
            value = -1
        if math.isnan(value):
            value = -1
        used_projection[i] = round(value,2)
    
    if used_target is None:
        return used_projection
    
    group_projection = {}
    for i in xrange(len(used_projection)):
        group = used_target[i]
        if not group in group_projection:
            group_projection[group]=[]
        group_projection[group].append(used_projection[i])
    return group_projection

def projection_to_histogram(projection):
    '''
    Return histogram of the projection in dictionary form.
    If the projection is list, then key of the dictionary will be every value in the projection, and dictionary's value
    
    Usage:
    >>> projection = [3,4,6,6]
    >>> hist = projection_to_histogram(projection)
    {3:1, 4:1, 6:2}
    >>>
    >>> projection = {'small':[3,4],'big':[6,6]}
    >>> hist = projection_to_histogram(projection)
    { 'small':{3:1, 4:1}, 'big':{6:2} }
    '''
    if isinstance(projection, dict):
        group_histogram = {}
        for group in projection:
            group_histogram[group] = projection_to_histogram(projection[group])
        return group_histogram
    else:
        histogram = {}
        for value in projection:
            if not value in histogram:
                histogram[value] = 0.0
            histogram[value]+=1
        return histogram

def get_group(per_group_dict):
    '''
    return keys of the dictionary
    
    Usage:
    >>> projection = {'small':[3,4],'big':[6,6]}
    >>> groups = get_group(projection)
    >>> print groups
    ['small', 'big']
    >>>
    >>> hist = { 'small':{3:1, 4:1}, 'big':{6:2} }
    >>> groups = get_group(hist)
    >>> print hist
    ['small', 'big']
    '''
    keys = []
    for key in per_group_dict:
        keys.append(key)
    return keys

def merge_projection(per_group_projection):
    '''
    Merge per_group_projection into total projection
    
    Usage:
    >>> projection = {'small':[3,4],'big':[6,6]}
    >>> total_projection = merge_projection(projection)
    >>> print total_projection
    [3,4,6,6]
    '''
    total_projection = []
    for group in get_group(per_group_projection):
        projection = per_group_projection[group]
        total_projection += projection
    return total_projection

def merge_histogram(per_group_hist):
    '''
    Merge per_group_hist into total histogram
    
    Usage:
    >>> hist = { 'small':{3:1, 4:1}, 'big':{6:2} }
    >>> total_hist = merge_hist(hist)
    >>> print total_hist
    {3:1, 4:1, 6:2}
    '''
    total_hist = {}
    for group in get_group(per_group_hist):
        hist = per_group_hist[group]
        for value in hist:
            if not value in total_hist:
                total_hist[value] = hist[value]
            else:
                total_hist[value] += hist[value]
    return total_hist

def calculate_mean(projection):
    '''
    Return mean of the projection.
    If the projection is list, then a scalar float value will be returned
    If the projection is dictionary, then a dictionary with the same key, and scalar float value will be returned
    
    Usage:
    >>> projection = [3,4,6,6]
    >>> mean = calculate_mean(projection)
    4.75
    >>>
    >>> projection = {'small':[3,4],'big':[6,6]}
    >>> mean = calculate_mean(projection)
    { 'small':3.5}, 'big':6 }
    '''
    if isinstance(projection, dict):
        group_mean = {}
        for group in projection:
            group_mean[group] = calculate_mean(projection[group])
        return group_mean
    else:
        mean = numpy.mean(projection)
        return mean

def calculate_std(projection):
    '''
    Return standard deviation of the projection.
    If the projection is list, then a scalar float value will be returned
    If the projection is dictionary, then a dictionary with the same key, and scalar float value will be returned
    
    Usage:
    >>> projection = [3,4,6,6]
    >>> mean = calculate_std(projection)
    1.299038105676658
    >>>
    >>> projection = {'small':[3,4],'big':[6,6]}
    >>> mean = calculate_std(projection)
    { 'small':0.5}, 'big':0.0 }
    '''
    if isinstance(projection, dict):
        group_std = {}
        for group in projection:
            group_std[group] = calculate_std(projection[group])
        return group_std
    else:
        std = numpy.std(projection)
        return std
    
def calculate_count(projection):
    '''
    Return data count in the projection.
    If the projection is list, then a scalar int value will be returned
    If the projection is dictionary, then a dictionary with the same key, and scalar int value will be returned
    
    Usage:
    >>> projection = [3,4,6,6]
    >>> count = calculate_count(projection)
    4
    >>>
    >>> projection = {'small':[3,4],'big':[6,6]}
    >>> mean = calculate_count(projection)
    { 'small':2}, 'big':2 }
    '''
    if isinstance(projection, dict):
        group_count = {}
        for group in projection:
            group_count[group] = calculate_count(projection[group])
        return group_count
    else:
        count = len(projection)
        return count

def calculate_separability_index(group_projection):
    '''
    This will return a dictionary contains costumized separability index of every group in group_projection
    
    The customized separability index 
    '''
    projection_group_count = calculate_count(group_projection)
    group_hist = projection_to_histogram(group_projection)
    total_hist = merge_histogram(group_hist)
    total_projection = merge_projection(group_projection)
    separability_index = {}
    for current_group in group_hist:
        # get current_hist and other_hist
        current_hist = group_hist[current_group]
        other_hist = dict(group_hist)
        del other_hist[current_group]
        other_hist = merge_histogram(other_hist)
        # get current and other's total_data_count
        current_total_data_count = projection_group_count[current_group]
        other_total_data_count = len(total_projection) - current_total_data_count
        current_separability_accumulation = 0.0
        other_separability_accumulation = 0.0
        
        # separability accumulation of current_hist
        for current_value in current_hist:
            total_count = 0
            good_count = 0
            distances = []
            count_on_distance = {}
            for compare_value in total_hist:
                distance = abs(compare_value - current_value)
                if not distance in distances:
                    distances.append(distance)
                    count_on_distance[distance] = {'good':0, 'all':0}
                count_on_distance[distance]['all'] += total_hist[compare_value]
                if compare_value in current_hist:
                    count_on_distance[distance]['good'] += current_hist[compare_value]
            distances.sort()
            for distance in distances:
                good_count += count_on_distance[distance]['good']
                total_count += count_on_distance[distance]['all']
                if good_count >= current_total_data_count:
                    break
            point_separability = current_hist[current_value]*(good_count-1)/max(total_count-1, LIMIT_ZERO)
            current_separability_accumulation += point_separability
        
        # separability accumulation of other_hist
        for other_value in other_hist:
            total_count = 0
            good_count = 0
            distances = []
            count_on_distance = {}
            for compare_value in total_hist:
                distance = abs(compare_value - other_value)
                if not distance in distances:
                    distances.append(distance)
                    count_on_distance[distance] = {'good':0, 'all':0}
                count_on_distance[distance]['all'] += total_hist[compare_value]
                if compare_value in other_hist:
                    count_on_distance[distance]['good'] += other_hist[compare_value]
            distances.sort()
            for distance in distances:                
                good_count += count_on_distance[distance]['good']
                total_count += count_on_distance[distance]['all']
                if good_count >= other_total_data_count:
                    break
            point_separability = other_hist[other_value]*(good_count-1)/max(total_count-1, LIMIT_ZERO)
            other_separability_accumulation += point_separability 
            
        current_separability_index = current_separability_accumulation/current_total_data_count
        other_separability_index = other_separability_accumulation/other_total_data_count
        min_current_separability_index = (current_total_data_count-1.0)/max((current_total_data_count+other_total_data_count-1.0),LIMIT_ZERO)
        min_other_separability_index = (other_total_data_count-1.0)/max((current_total_data_count+other_total_data_count-1.0),LIMIT_ZERO)
        normalized_current_separability_index = (current_separability_index-min_current_separability_index)/(1-min_current_separability_index)
        normalized_other_separability_index = (other_separability_index-min_other_separability_index)/(1-min_other_separability_index)
        separability_index[current_group] = min(normalized_current_separability_index,normalized_other_separability_index)
    return separability_index

def calculate_collision_proportion(group_projection):
    '''
    This will return a dictionary contains collision proportion of every group in group_projection
    
    The calculation is as follows:
    Calculate summary_of_all_data in collision point/data_count
    '''
    #total_projection = merge_projection(group_projection)
    #data_count = len(total_projection)
    group_hist = projection_to_histogram(group_projection)
    total_hist = merge_histogram(group_hist)    
    total_projection = merge_projection(group_projection)
    data_count = len(total_projection)
    collision_proportion = {}
    for group in group_hist:        
        collision_count = 0.0
        other_hist = dict(group_hist)
        del other_hist[group]
        other_hist = merge_histogram(other_hist)
        current_hist = group_hist[group]
        for value in current_hist:            
            for compare_value in other_hist:
                if compare_value==value:
                    collision_count += total_hist[compare_value]
        collision_proportion[group] = collision_count/data_count
    return collision_proportion

def calculate_intrusion_proportion(group_projection):
    total_projection = merge_projection(group_projection)
    #data_count = len(total_projection)
    group_hist = projection_to_histogram(group_projection)
    total_hist = projection_to_histogram(total_projection)
    total_projection = merge_projection(group_projection)
    data_count = len(total_projection)    
    intrusion_proportion = {}
    for group in group_projection:
        current_projection = group_projection[group]
        other_hist = dict(group_hist)
        del other_hist[group]
        other_hist = merge_histogram(other_hist)
        other_projection = dict(group_projection)
        del other_projection[group]
        other_projection = merge_projection(other_projection)
        min_other = min(other_projection)
        min_current = min(current_projection)
        max_other = max(other_projection)
        max_current = max(current_projection)
        maximum_min = max(min_other, min_current)
        minimum_max = min(max_other, max_current)
        # calculate intrusion count
        intrussion_count = 0.0
        for value in total_hist:
            if value>=maximum_min and value>=min_current and value<=minimum_max and value<=max_current:
                intrussion_count += total_hist[value]
        intrusion_proportion[group] = intrussion_count/data_count
    return intrusion_proportion

def calculate_distance(group_projection):  
    distance = {}
    for group in group_projection:
        current_projection = group_projection[group]
        other_projection = dict(group_projection)
        del other_projection[group]
        other_projection = merge_projection(other_projection)
        current_mean = calculate_mean(current_projection)
        other_mean = calculate_mean(other_projection)
        distance[group] = abs(current_mean-other_mean)
    return distance

def calculate_max_accuration_prediction(group_projection):
    '''
    This will return a dictionary contains max_accuration of every group in group_projection
    
    The calculation is as follows:
    For a range in minimum margin to maximum margin of each group_projection
    Choose all possibility of min_val and max_val where max_val>=min_val
    For every possibility, calculate the miss-classification_count, and choose the most minimum miss-classification_count
    Where:
    
    miss-classification_count = data count of current group below min_val + data count of current_group above max_val +
        data count of other group between min_val and max_val
    
    max_accuration = minimum_miss-classification_count / data_count
    
    '''
    total_projection = merge_projection(group_projection)
    data_count = len(total_projection)
    group_hist = projection_to_histogram(group_projection)
    total_hist = merge_histogram(group_hist)
    all_values = get_group(total_hist)
    accuration = {}
    for group in group_hist:
        min_miss_count = 0.0
        group_values = get_group(group_hist[group])
        group_values.sort()
        for i in xrange(len(group_values)): # min_val loop
            min_val = group_values[i]-LIMIT_ZERO
            for j in xrange(i,len(group_values)): # max_val loop
                max_val = group_values[j]+LIMIT_ZERO
                miss_count = 0.0
                # group values
                for k in xrange(len(group_values)): # all value loop.
                    current_val = group_values[k]
                    all_count = total_hist[current_val]
                    current_count = group_hist[group][current_val]
                    if current_val<min_val: # less than min_val
                        miss_count += current_count
                    elif current_val<=max_val: # between min_val and max_val
                        miss_count += all_count - current_count
                    else: # more than max_val
                        miss_count += current_count
                # all other values which is not in group values but located between min_val and max_val
                for value in all_values:
                    if (not value in group_values) and (value>=min_val) and (value<=max_val):
                        count = total_hist[value]
                        miss_count += count
                if (i==0 and j==0) or (min_miss_count>miss_count):
                    min_miss_count = miss_count
        accuration[group] = (data_count-min_miss_count)/float(data_count)
    return accuration

def calculate_f_score(group_projection):
    '''
    calculate fscore
    '''
    total_projection = merge_projection(group_projection)
    total_mean = calculate_mean(total_projection)
    f_score = {}
    for label in group_projection:
        current_projection = group_projection[label]
        compare_projection = dict(group_projection)
        del compare_projection[label]
        compare_projection = merge_projection(compare_projection)
        current_mean = calculate_mean(current_projection)
        current_data_count = calculate_count(current_projection)
        compare_mean = calculate_mean(compare_projection)
        compare_data_count = calculate_count(compare_projection)
        current_accumulation = 0.0
        compare_accumulation = 0.0
        for val in current_projection:
            current_accumulation += (val-current_mean)**2
        for val in compare_projection:
            compare_accumulation += (val-compare_mean)**2
        f_score[label] = ( (current_mean-total_mean)**2 + (compare_mean-total_mean)**2 ) 
        f_score[label]/= max(LIMIT_ZERO, ( current_accumulation/max(LIMIT_ZERO,(current_data_count-1)) + compare_accumulation/max(LIMIT_ZERO,(compare_data_count-1)) ))
    return f_score

def calculate_linear_max_accuration_prediction(group_projection):
    total_projection = merge_projection(group_projection)
    total_count = len(total_projection)
    group_hist = projection_to_histogram(group_projection)
    total_hist = projection_to_histogram(total_projection)
    accuracy = {}
    for group in group_projection:
        current_hist = group_hist[group]
        other_hist = dict(group_hist)
        del other_hist[group]
        other_hist = merge_histogram(other_hist)
        max_accuracy = 0
        for margin in total_hist:
            left_current_count = 0
            right_current_count = 0
            left_other_count = 0
            right_other_count = 0
            for value in total_hist:
                if value<=margin+LIMIT_ZERO:
                    if value in current_hist:
                        left_current_count += current_hist[value]
                    if value in other_hist:
                        left_other_count += other_hist[value]
                else:
                    if value in current_hist:
                        right_current_count += current_hist[value]
                    if value in other_hist:
                        right_other_count += other_hist[value]
                left_accuracy = (left_current_count+right_other_count)/total_count
                right_accuracy = (right_current_count+left_other_count)/total_count
                max_accuracy = max(left_accuracy, right_accuracy, max_accuracy)
        accuracy[group] = max_accuracy
    return accuracy

def calculate_class_purity(group_projection):
    '''
    This will return the proportion of class data/total_data in class range
    '''
    group_hist = projection_to_histogram(group_projection)
    total_hist = merge_histogram(group_hist)
    class_purity = {}
    for group in group_hist:
        current_projection = group_projection[group]
        min_value = min(current_projection)
        max_value = max(current_projection)
        current_count = 0.0
        total_count = 0.0
        for value in total_hist:
            if value>=min_value-LIMIT_ZERO and value<=max_value+LIMIT_ZERO:
                total_count += total_hist[value]
                if value in group_hist[group]:
                    current_count += group_hist[group][value]
        class_purity[group] = current_count/total_count
    return class_purity

def my_metric(group_projection):
    separability_indexes = calculate_separability_index(group_projection)
    #collision_proportions = calculate_collision_proportion(group_projection)
    #intrusion_proportions = calculate_intrusion_proportion(group_projection)
    #distances = calculate_distance(group_projection)
    result = {}
    for group in group_projection:
        #distance = distances[group]
        separability_index = separability_indexes[group]
        #intrusion_proportion = intrusion_proportions[group]
        #collision_proportion = collision_proportions[group]
        #distance = min(distance,1)
        '''
        value = ( 3*separability_index + 2*(1-intrusion_proportion)*separability_index )/5
        if collision_proportion>0:
            value = (value + 7*(1-collision_proportion)*separability_index)/8
        value = (3*value + 2*distance)/5
        '''
        value = separability_index
        result[group] = max(0, value)
    return result

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
        self.target_dictionary = {}
        group_index = 0
        for i in xrange(len(records)):
            record = records[i]
            if i==0:
                self.features = record[:-1]
            else:
                self.data.append(record[:-1])
                self.label_target.append(record[-1])
                if not (record[-1] in self.target_dictionary):
                    self.target_dictionary[record[-1]] = group_index
                    self.group_count[record[-1]] = 0
                    self.group_label.append(record[-1])              
                    group_index += 1
                self.group_count[record[-1]] += 1           
                self.num_target.append(self.target_dictionary[record[-1]])                
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
        
class Genetics_Feature_Extractor(Feature_Extractor, classes.GA_Base):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        Feature_Extractor.__init__(self, records, fold_count, fold_index)
        classes.GA_Base.__init__(self)
        if classifier is None:
            try:
                self.classifier = svm.SVC(max_iter=2000, class_weight='auto')
                #self.classifier = GaussianNB()
            except:
                self.classifier = svm.SVC(class_weight='auto')
                #self.classifier = GaussianNB()
        else:
            self.classifier = classifier
        
    def get_new_features(self):
        return self.features
    
    def get_metrics(self, new_feature):
        group_projection = get_projection(new_feature, self.features, self.data, self.training_data, self.training_label_target)        
        group_hist = projection_to_histogram(group_projection)
        total_hist = merge_histogram(group_hist)
        max_accuration_prediction = calculate_max_accuration_prediction(group_projection)
        linear_max_accuration_prediction = calculate_linear_max_accuration_prediction(group_projection)
        collision_proportion = calculate_collision_proportion(group_projection)
        intrusion_proportion = calculate_intrusion_proportion(group_projection)
        separability_index = calculate_separability_index(group_projection)
        mean = calculate_mean(group_projection)
        stdev = calculate_std(group_projection)
        distance = calculate_distance(group_projection)
        
        # return metric
        metric = {
            'total_histogram': total_hist,
            'group_histogram': group_hist,
            'max_accuration_prediction' : max_accuration_prediction,
            'linear_max_accuration_prediction' : linear_max_accuration_prediction,
            'collision_proportion' : collision_proportion,
            'intrusion_proportion' : intrusion_proportion,
            'separability_index':separability_index,
            'mean':mean,
            'stdev':stdev,
            'distance':distance            
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
        # zero matrix
        for i in xrange(len(training_data)):
            new_training_data.append([0]*new_feature_count)
        for i in xrange(len(test_data)):
            new_test_data.append([0]*new_feature_count)
        for i in xrange(new_feature_count):
            feature = new_features[i]
            training_projection = get_projection(feature, self.features, self.data, training_data)
            test_projection = get_projection(feature, self.features, self.data, test_data)
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
        accuracy = {'training':training_accuracy, 'test':test_accuracy, 'total':total_accuracy}
        return accuracy

class GA_Select_Feature(Genetics_Feature_Extractor, classes.Genetics_Algorithm):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        Genetics_Feature_Extractor.__init__(self, records, fold_count, fold_index, classifier)     
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
        Genetics_Feature_Extractor.__init__(self, records, fold_count, fold_index, classifier)     
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
        Genetics_Feature_Extractor.__init__(self, records, fold_count, fold_index, classifier)     
        classes.Grammatical_Evolution.__init__(self)
        self.fitness_measurement = 'MAX'
        self.benchmarks = ['accuration']
        self.variables = self.features
        self.grammar = {
            '<expr>' : ['<var>','(<expr>) <op> (<expr>)','<func>(<expr>)'],
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

class Multi_Accuration_Fitness(Genetics_Feature_Extractor):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        Genetics_Feature_Extractor.__init__(self, records, fold_count, fold_index, classifier)
        self.benchmarks = self.group_label
    
    def get_new_features(self):
        new_features = []
        for benchmark in self.benchmarks:
            best_phenotype = self.best_individuals(1, benchmark, 'phenotype')
            if best_phenotype not in new_features:
                new_features.append(best_phenotype)
        return new_features
    
    def do_calculate_fitness(self, individual):
        feature = individual['phenotype']
        total_projection = get_projection(feature, self.features, self.data, self.training_data)
        new_training_data = []
        for i in xrange(len(total_projection)):
            new_training_data.append([total_projection[i]])
        
        fitness = {}
        for group in self.group_label:
            group_index = self.target_dictionary[group]
            new_target = list(self.training_num_target)
            '''
            for i in xrange(len(new_target)):
                if not new_target[i] == group_index:
                    new_target[i] = -1
            '''
            self.classifier.fit(new_training_data, new_target)
            prediction = self.classifier.predict(new_training_data)
            true_count = 0.0
            false_count = 0.0
            for i in xrange(len(prediction)):
                if prediction[i] == group_index or new_target[i] == group_index:
                    if prediction[i] == new_target[i]:
                        true_count += 1
                    else:
                        false_count += 1
            accuration = true_count/(true_count+false_count)
            fitness[group] = accuration        
        return fitness

class GE_Multi_Accuration_Fitness(GE_Select_Feature, Multi_Accuration_Fitness):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GE_Select_Feature.__init__(self, records, fold_count, fold_index, classifier)
        Multi_Accuration_Fitness.__init__(self, records, fold_count, fold_index, self.classifier)
    
    def do_calculate_fitness(self, individual):
        return Multi_Accuration_Fitness.do_calculate_fitness(self, individual)
    
    def get_new_features(self):
        return Multi_Accuration_Fitness.get_new_features(self)    
    
class GP_Multi_Accuration_Fitness(GP_Select_Feature, Multi_Accuration_Fitness):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GP_Select_Feature.__init__(self, records, fold_count, fold_index, classifier)
        Multi_Accuration_Fitness.__init__(self, records, fold_count, fold_index, self.classifier)
    
    def do_calculate_fitness(self, individual):
        return Multi_Accuration_Fitness.do_calculate_fitness(self, individual)
    
    def get_new_features(self):
        return Multi_Accuration_Fitness.get_new_features(self)

class Global_Separability_Fitness(Genetics_Feature_Extractor):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        self.benchmarks = ['separability']
    
    def get_new_features(self):
        best_phenotype = self.best_individuals(1, 'separability', 'phenotype')
        return [best_phenotype]
    
    def do_calculate_fitness(self, individual):
        new_feature = individual['phenotype']
        group_projection = get_projection(new_feature, self.features, self.data, self.training_data, self.training_label_target)
        local_fitness = my_metric(group_projection)
        global_fitness = 0.0
        for benchmark in local_fitness:
            global_fitness += local_fitness[benchmark]
        global_fitness /= len(local_fitness)
        
        fitness = {}
        fitness['separability'] = global_fitness
        return fitness

class Local_Separability_Fitness(Genetics_Feature_Extractor):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        self.benchmarks = self.group_label
    
    def get_new_features(self):
        new_features = []
        for benchmark in self.benchmarks:            
            best_phenotype = self.best_individuals(1, benchmark, 'phenotype')
            if best_phenotype not in new_features:
                new_features.append(best_phenotype)
        return new_features
    
    def do_calculate_fitness(self, individual):
        new_feature = individual['phenotype']
        group_projection = get_projection(new_feature, self.features, self.data, self.training_data, self.training_label_target)
        fitness = my_metric(group_projection)
        return fitness

class GP_Global_Separability_Fitness(GP_Select_Feature, Global_Separability_Fitness):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GP_Select_Feature.__init__(self, records, fold_count, fold_index, classifier)
        Global_Separability_Fitness.__init__(self, records, fold_count, fold_index, classifier)
    
    def do_calculate_fitness(self, individual):
        return Global_Separability_Fitness.do_calculate_fitness(self, individual)
    
    def get_new_features(self):
        return Global_Separability_Fitness.get_new_features(self)

class GP_Local_Separability_Fitness(GP_Select_Feature, Local_Separability_Fitness):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GP_Select_Feature.__init__(self, records, fold_count, fold_index, classifier)
        Local_Separability_Fitness.__init__(self, records, fold_count, fold_index, classifier)
    
    def do_calculate_fitness(self, individual):
        return Local_Separability_Fitness.do_calculate_fitness(self, individual)
    
    def get_new_features(self):
        return Local_Separability_Fitness.get_new_features(self)

class GE_Global_Separability_Fitness(GE_Select_Feature, Global_Separability_Fitness):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GE_Select_Feature.__init__(self, records, fold_count, fold_index, classifier)
        Global_Separability_Fitness.__init__(self, records, fold_count, fold_index, classifier)
    
    def do_calculate_fitness(self, individual):
        return Global_Separability_Fitness.do_calculate_fitness(self, individual)
    
    def get_new_features(self):
        return Global_Separability_Fitness.get_new_features(self)

class GE_Local_Separability_Fitness(GE_Select_Feature, Local_Separability_Fitness):
    def __init__(self, records, fold_count=1, fold_index=0, classifier=None):
        GE_Select_Feature.__init__(self, records, fold_count, fold_index, classifier)
        Local_Separability_Fitness.__init__(self, records, fold_count, fold_index, classifier)
    
    def do_calculate_fitness(self, individual):
        return Local_Separability_Fitness.do_calculate_fitness(self, individual)
    
    def get_new_features(self):
        return Local_Separability_Fitness.get_new_features(self)
        
def extract_feature(records, data_label='Test', fold_count=5, extractors=[], classifier=None):
    if extractors is None or len(extractors) == 0:
        extractors = [
            {'class': GA_Select_Feature, 'label':'GA', 'color':'red', 'params':{}},
            {'class': GP_Select_Feature, 'label':'GP', 'color':'orange', 'params':{'max_epoch':100,'population_size':200}},
            {'class': GP_Global_Separability_Fitness, 'label':'GP Global', 'color':'green', 'params':{'max_epoch':100,'population_size':200}},
            {'class': GP_Local_Separability_Fitness, 'label':'GP Local', 'color':'blue', 'params':{'max_epoch':100,'population_size':200}},
            {'class': GE_Select_Feature, 'label':'GE', 'color':'cyan', 'params':{'max_epoch':100,'population_size':200}},
            {'class': GE_Global_Separability_Fitness, 'label':'GE Global', 'color':'magenta', 'params':{'max_epoch':100,'population_size':200}},
            {'class': GE_Local_Separability_Fitness, 'label':'GE Local', 'color':'black', 'params':{'max_epoch':100,'population_size':200}},
            {'class': GP_Multi_Accuration_Fitness, 'label':'GP Multi', 'color':'magenta', 'params':{'max_epoch':100,'population_size':200}},
            {'class': GE_Multi_Accuration_Fitness, 'label':'GE Multi', 'color':'magenta', 'params':{'max_epoch':100,'population_size':200}}
        ]
    
    # delete invalid extractors
    i = 0
    while i<len(extractors):
        is_good_extractor = True
        extractor = extractors[i]
        if not ( ('class' in extractor) and ('label' in extractor) and ('color' in extractor) ):
            is_good_extractor = False
        if not is_good_extractor:
            del extractors[i]
        else:
            i+=1

        
    
    if len(extractors) == 0:
        print 'No extractor set ...'
        return 0
    
    shown_metrics = {
        'max_accuration_prediction' : 'Maximum Accuration Prediction',
        'collision_proportion' : 'Collision Proportion',
        'intrusion_proportion' : 'Intrusion Proportion',
        'separability_index': 'Separability Index',
        'mean': 'Means',
        'stdev': 'Standard Deviation',
        'linear_max_accuration_prediction': 'Linear Maximum Accuration Prediction',
        'distance':'Distance'
    }
        
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
            # add params to extractor
            if 'params' in extractor:
                for param_name in extractor['params']:
                    param_value = extractor['params'][param_name]
                    try:
                        exec('fe.'+param_name+' = '+str(param_value))
                    except:
                        print 'failed to assign '+param_name+'='+str(param_value)+' of '+extractor_label
                        pass
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
                total_histogram[feature] = metrics['total_histogram']
                group_histogram[feature] = metrics['group_histogram']
                # show shown metrics
                for key in shown_metrics:
                    label_metric = shown_metrics[key]                
                    output += '    '+label_metric+' :\r\n'
                    for group in groups:
                        label_group = group
                        while len(label_group)<max_group_label_length:
                            label_group += ' '
                        output += '      '+label_group+' : '+str(metrics[key][group])+'\r\n'
                                        
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
                        sp.plot([value,value],[group_index, group_index+0.9*count/total_max_count],'k', linewidth=1)
                    sp.plot([group_min_value, group_max_value],[group_index, group_index],'k')
                    group_index += 1                      
                
                
                # plot global histogram
                sp.plot([total_min_value,total_max_value],[0,0],'k')
                for value in feature_total_histogram:
                    count = feature_total_histogram[value]
                    sp.plot([value,value], [0, 0.9*count/total_max_count],'k', linewidth=1)                
                
                    
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
            
    
            plt.savefig(data_label+'/'+extractor_label+' Fold '+str(fold_index+1)+' Feature Projection.png', dpi=100)
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
        helper_value_drawn = []
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
                label = extractors[extractor_index]['label']
                value = all_accuracy[label][metric_label][fold_index]
                if not value in helper_value_drawn:
                    helper_value_drawn.append(value)
                    sp.plot([min_x, max_x],[value*100,value*100],'k--')
                
        sp.set_ylim(minimum_accuracy*100,maximum_accuracy*100)
        sp.set_xlim(min_x, max_x)
        sp.set_title(metric_label.capitalize())
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

def measure_metrics(records, metric_measurement=None, classifier = None, measurement_label='test'):
    if metric_measurement is None:
        metric_measurement = {
            'means': calculate_mean,
            'standard deviation': calculate_std,
            'separability index': calculate_separability_index,
            'collision proportion':calculate_collision_proportion,
            'max accuration prediction':calculate_max_accuration_prediction,
            'class purity':calculate_class_purity
        }
    
    if classifier is None:
        classifier = svm.SVC()
        #classifier = GaussianNB()
    
    # prepare feature extractor
    fe = GE_Select_Feature(records, fold_count=1, fold_index=0)
    fe.population_size = 500
    fe.max_epoch = 5
    fe.elitism_rate = 0.01
    fe.new_rate = 0.6
    fe.mutation_rate = 0.3
    fe.crossover_rate = 0.29
    
    # get individuals
    individuals = []
    file_name = measurement_label+' individuals.p'
    if os.path.isfile(file_name):
        individuals = pickle.load(open(file_name,"rb"))
    else:
        fe.process()    
        gp = GP_Select_Feature(records, fold_count=1, fold_index=0)
        gp.population_size = 500
        gp.max_epoch = 5
        gp.process()
        individuals = fe._individuals + gp._individuals
        pickle.dump(individuals, open(file_name, "wb"))
    
    
    # get parameters
    data = fe.training_data
    num_target = fe.training_num_target
    label_target = fe.training_label_target
    target_dictionary = fe.target_dictionary    
    old_features = fe.features
    metric_values = []
    accurations = []
    individual_index = 0
    for individual in individuals:        
        feature = individual['phenotype']
        # get total and group projection
        total_projection = get_projection(feature, old_features, all_data=data, used_data=data)
        group_projection = get_projection(feature, old_features, all_data=data, used_data=data, used_target=label_target)
        # get metric value
        metric_value = {}
        for measurement in metric_measurement:
            metric_value[measurement] = metric_measurement[measurement](group_projection)
        # build new data
        new_data = []
        for projection_value in total_projection:
            new_data.append([projection_value])
        # train and predict svm
        classifier.fit(new_data,num_target)
        prediction = classifier.predict(new_data)
        # calculate accuration
        accuration = {}        
        for group in target_dictionary:
            group_index = target_dictionary[group]
            count = 0.0
            miss_count = 0.0
            for i in xrange(len(num_target)):
                if num_target[i] == group_index or prediction[i] == group_index:
                    if num_target[i] <> prediction[i]:
                        miss_count += 1
                    count += 1
            accuration[group] = (count-miss_count)/max(count,LIMIT_ZERO)
        
        metric_values.append(metric_value)
        accurations.append(accuration)
        
        individual_index += 1
        utils.write('calculate feature '+str(individual_index)+' of '+str(len(individuals)))
    
    # draw graphics
    group_count = len(fe.group_label)
    metric_count = len(metric_measurement)
    feature_count = len(accurations)
    fig = plt.figure(figsize=(group_count*5.0, metric_count*5.0))
    
    metric_index = 0
    for metric in metric_measurement:
        for group in target_dictionary:
            group_index = target_dictionary[group]
            sp = fig.add_subplot(metric_count, group_count, (metric_index*group_count+group_index)+1)
            x_arr = []
            y_arr = []
            for feature_index in xrange(feature_count):
                x = metric_values[feature_index][metric][group]
                y = accurations[feature_index][group]
                x_arr.append(x),
                y_arr.append(y)
                sp.plot(x,y,'bo', markersize=2.0)
            correlation, p_value = pearsonr(x_arr,y_arr)            
            sp.set_title(group+'\n Corr : '+str(correlation)+'\n P-Value : '+str(p_value))
            sp.set_ylim(-0.01, 1.01)
            sp.set_xlim(-0.01, max(1.01, max(x_arr)))
            sp.set_ylabel('Accuration')
            sp.set_xlabel(metric)
            utils.write('Plotting '+metric+' and accuration correlation of '+group)
        metric_index += 1
    print '\nDraw Plot ...'    
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    plt.suptitle('Metric and Accuracy Correlation')
    plt.savefig('Metric Measurement '+measurement_label+'.png', dpi=100)
    fig.clf()
    plt.close()
    gc.collect()
    print 'Done ...'
