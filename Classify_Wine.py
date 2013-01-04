from Feature_Extractor import *

records = extract_csv('wine.data.csv', delimiter=',')
records = shuffle_record(records)
fold_count = 5
data_label = 'Wine Data'
extractors = None
extract_feature(records, 'Wine - Normal SVM', fold_count, extractors)
extract_feature(records, 'Wine - Linear SVM', fold_count, extractors, svm.SVC(kernel='linear'))