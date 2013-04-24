import Feature_Extractor as FE
from sklearn.naive_bayes import GaussianNB
import sys

def test(csv_file_name, new_feature):
    records = FE.extract_csv(csv_file_name, delimiter=',')
    records = FE.shuffle_record(records)
    # preprocess records
    data = []
    label_target = []
    num_target = []
    group_label = []
    group_count = {}
    target_dictionary = {}
    group_index = 0
    features = []
    for i in xrange(len(records)):
        record = records[i]
        if i==0:
            features = record[:-1]
        else:
            data.append(record[:-1])
            label_target.append(record[-1])
            if not (record[-1] in target_dictionary):
                target_dictionary[record[-1]] = group_index
                group_count[record[-1]] = 0
                group_label.append(record[-1])              
                group_index += 1
            group_count[record[-1]] += 1           
            num_target.append(target_dictionary[record[-1]])                
    group_label.sort()
    # calculate projection and prediction    
    projection = FE.get_projection(new_feature, features, data)
    projection_data = []
    for projection_value in projection:
        projection_data.append([projection_value])
    target = num_target
    classifier = GaussianNB()
    prediction = classifier.fit(projection_data, target).predict(projection_data)
    # initiate true_count, false_count and accuracy
    true_count = dict(target_dictionary)
    false_count = dict(target_dictionary)
    accuracy = dict(target_dictionary)
    # reverse target dictionary
    reverse_target_dictionary = {}
    for key in target_dictionary:
        val = target_dictionary[key]
        reverse_target_dictionary[val] = key
    for key in true_count:
        true_count[key] = 0.0
        false_count[key] = 0.0
        accuracy[key] = 0.0
    # calculate true_count and false_count
    for i in xrange(len(target)):
        target_value = target[i]
        prediction_value = prediction[i]
        if target_value == prediction_value:
            true_count[reverse_target_dictionary[prediction_value]] += 1
        else:
            false_count[reverse_target_dictionary[prediction_value]] += 1
    # calculate accuracy
    for key in accuracy:
        accuracy[key] = true_count[key]/(true_count[key]+false_count[key])
    print(accuracy)
        
        


if __name__ == '__main__':
    if len(sys.argv)>1:
        csv_file_name = sys.argv[1]
    else:
        print('Give me a csv file name:')
        csv_file_name = raw_input()
    new_feature = raw_input('Give me new feature : ')
    test(csv_file_name, new_feature)