import svm
from Feature_Extractor import *

def the_metric(group_projection):
    separability_indexes = calculate_separability_index(group_projection)
    collision_proportions = calculate_collision_proportion(group_projection)
    intrusion_proportions = calculate_intrusion_proportion(group_projection)
    #distances = calculate_distance(group_projection)
    result = {}
    for group in group_projection:
        #distance = distances[group]
        separability_index = separability_indexes[group]
        intrusion_proportion = intrusion_proportions[group]
        collision_proportion = collision_proportions[group]
        #distance = min(distance,1)
        value = (3*separability_index + 2*(1-collision_proportion)*separability_index + (1-intrusion_proportion)*separability_index)/6.0
        result[group] = max(0, value)
    return result

if __name__ == '__main__':
    records = extract_csv('iris.data.csv', delimiter=',')
    new_records = []
    for i in xrange(len(records)):
        class_index = len(records[i])-1
        if not records[i][class_index] == 'Iris-virginica':
            new_records.append(records[i])
    records = new_records
    
    metric_measurement = {
        'separability index': calculate_separability_index,
        'collision proportion':calculate_collision_proportion,
        'intrussion proportion':calculate_intrusion_proportion,
        'max accuration prediction':calculate_max_accuration_prediction,
        'max linear accuration prediction':calculate_linear_max_accuration_prediction,
        'f_score':calculate_f_score,
        'distance':calculate_distance,
        'the metric': the_metric
    }
    #measure_metrics(records, metric_measurement, measurement_label='Normal Kernel')
    svc = svm.SVC(kernel='linear')
    measure_metrics(records, metric_measurement, measurement_label='Linear Kernel', classifier=svc)