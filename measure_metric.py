import svm
from Feature_Extractor import *

if __name__ == '__main__':
    records = extract_csv('iris.data.csv', delimiter=',')
    for i in xrange(len(records)):
        class_index = len(records[i])-1
        if not records[i][class_index] == 'Iris-setosa':
            records[i][class_index] = 'other' 
    metric_measurement = {
        'separability index': calculate_separability_index,
        'collision proportion':calculate_collision_proportion,
        'intrussion proportion':calculate_intrusion_proportion,
        'max accuration prediction':calculate_max_accuration_prediction,
        'max linear accuration prediction':calculate_linear_max_accuration_prediction,
        'f_score':calculate_f_score,
        'distance':calculate_distance,
        'my metric': my_metric
    }
    #measure_metrics(records, metric_measurement, measurement_label='Normal Kernel')
    svc = svm.SVC(kernel='linear')
    measure_metrics(records, metric_measurement, measurement_label='Linear Kernel', classifier=svc)