from FE_Base import Feature_Extractor
from gogenpy import utils
    
randomizer = utils.randomizer
records = []
for i in xrange(100):
    c = 'class 1';
    x = 0
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        y = randomizer.randrange(1,5)
    else:
        y = randomizer.randrange(-4,0)
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        z = randomizer.randrange(1,5)
    else:
        z = randomizer.randrange(-4,0)
    records.append([x,y,z,c])

for i in xrange(100):
    c = 'class 2';
    y = 0
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        x = randomizer.randrange(1,5)
    else:
        x = randomizer.randrange(-4,0)
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        z = randomizer.randrange(1,5)
    else:
        z = randomizer.randrange(-4,0)
    records.append([x,y,z,c])

for i in xrange(100):
    c = 'class 3';
    z = 0
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        y = randomizer.randrange(1,5)
    else:
        y = randomizer.randrange(-4,0)
    rnd = randomizer.randrange(0,2)
    if rnd>=1:
        x = randomizer.randrange(1,5)
    else:
        x = randomizer.randrange(-4,0)
    records.append([x,y,z,c])

variables = ['x','y','z']


# make feature extractor
fe = Feature_Extractor()
fe.label = 'Ideal data'
fe.max_epoch = 200
fe.records = records
fe.fold = 10
fe.variables = variables
fe.measurement = 'error'
fe.process()

'''
from sklearn import svm
from FE_Base import get_svm_result

def label_to_num(label):
    if label == 'class 1':
        return 1
    elif label == 'class 2':
        return 2
    else:
        return 3

new_features = variables
old_features = variables
training_data = []
training_target = []
test_data = []
test_target = []
data_count_per_class = int(len(records)/3)
for i in xrange(3):
    training_data_count = 0
    for j in xrange(i*data_count_per_class, (i+1)*data_count_per_class):
        if training_data_count<90:
            training_data_count += 1
            training_data.append(records[j][:-1])
            training_target.append(label_to_num(records[j][-1]))
        else:
            test_data.append(records[j][:-1])
            test_target.append(label_to_num(records[j][-1]))

print training_data
print training_target
print test_data
print test_target

#svc = svm.SVC(kernel='poly', degree=2)
svc = svm.SVC(kernel='rbf', gamma=0.4)
result = get_svm_result(training_data, training_target, test_data, test_target, old_features, new_features, 'coba', svc)
print result['str']
'''