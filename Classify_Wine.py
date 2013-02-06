from Feature_Extractor import *

records = extract_csv('wine.data.csv', delimiter=',')
records = shuffle_record(records)

data_label = 'Wine Data'
extractors = None
fold_count = 1
extract_feature(records, 'Wine (whole) - Normal SVM', fold_count, extractors)
extract_feature(records, 'Wine (whole) - Linear SVM', fold_count, extractors, svm.SVC(kernel='linear'))
fold_count = 5
extract_feature(records, 'Wine (5 fold) - Normal SVM', fold_count, extractors)
extract_feature(records, 'Wine (5 fold) - Linear SVM', fold_count, extractors, svm.SVC(kernel='linear'))
