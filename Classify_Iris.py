from Feature_Extractor import *
from sklearn import svm

records = extract_csv('iris.data.csv', delimiter=',')
records = shuffle_record(records)
fold_count = 5
extractors = None
extract_feature(records, 'Iris - Normal SVM', fold_count, extractors)
extract_feature(records, 'Iris - Linear SVM', fold_count, extractors, svm.SVC(kernel='linear'))
fold_count = 1
extract_feature(records, 'Iris (whole) - Normal SVM', fold_count, extractors)
extract_feature(records, 'Iris (whole) - Linear SVM', fold_count, extractors, svm.SVC(kernel='linear'))