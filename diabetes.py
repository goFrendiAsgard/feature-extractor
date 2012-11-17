'''
Created on Nov 15, 2012

@author: gofrendi
'''
from sklearn import svm, datasets
import classes, utils

used_benchmark = 'error_count' # mse or error_count, choose one :)
ds = datasets.load_diabetes()
data = list(ds.data)
target = list(ds.target)

# only take 30 data for training
training_data = data[0:100]
training_target = target[0:100]
print training_target

# use all data as test set
test_data = data
test_target = target

class GA(classes.Genetics_Algorithm): 
    def __init__(self):
        super(GA,self).__init__()
        self.fitness_measurement = 'MIN'
        self.representations=['default','svc', 'error_count']
        self.benchmarks=['error_count', 'mse']
        self.individual_length=20
    
    def process(self):
        if not len(training_data) == 0 and not len(training_data[0]) == 0:
            self.individual_length = len(training_data[0])+6
        else:
            self.individual_length = 20
        classes.Genetics_Algorithm.process(self)
    
    def do_process_individual(self, individual):
        gene = individual['default']
        # if the dataset is empty, then everything is impossible, don't try to do anything
        if not len(training_data) == 0 and not len(training_data[0]) == 0:
            feature_count = len(data[0]) # how many feature available in the data
            new_training_data = [] # we will use new training data, it could has less feature than the original one
            for i in xrange(len(training_data)):
                new_record = []
                for j in xrange(feature_count):
                    if gene[j] == '1':
                        new_record.append(training_data[i][j])
                new_training_data.append(new_record)  
            # we've just get the new training_data
            # perform svm (actually we still able to give some improvements, like choosing kernel etc)
            kernel_gene = gene[feature_count:feature_count+2]
            degree_gene = gene[feature_count+2:feature_count+4]
            gamma_gene = gene[feature_count+4:feature_count+6]
            kernel_option = {'00':'linear', '01':'linear', '10':'rbf', '11':'poly'}
            degree_value = utils.bin_to_dec(degree_gene)+1
            gamma_value = float(utils.bin_to_dec(gamma_gene)+1)/10.0
            svc = svm.SVC(kernel=kernel_option[kernel_gene], C=1.0, degree=degree_value, gamma=gamma_value).fit(new_training_data, training_target)
            prediction = svc.predict(new_training_data)
            error_count = 0
            error_square_sum = 0
            for i in xrange(len(training_data)):
                if prediction[i] <> training_target[i]:
                    error_count += 1
                error_square_sum += (prediction[i] - training_target[i])**2
            mse = error_square_sum/len(training_data)
            individual['svc'] = svc
            individual['error_count'] = error_count
            individual['mse'] = mse
        else:
            individual['svc'] = None
            individual['error_count'] = 1000
            individual['mse'] = 1000
        return individual
           
    def do_calculate_fitness(self, individual):                    
        return {'error_count': individual['error_count'], 'mse':individual['mse']}

if __name__ == '__main__':
    # process genetics algorithm
    ga = GA()
    ga.max_epoch = 200
    ga.population_size = 100
    # ga.stopping_value = 0
    ga.process()
    
    
    # prepare svm created by genetics algorithm
    gene = ga.best_individuals(1, representation='default', benchmark=used_benchmark)
    new_test_data = [] # we will use new test data, it could has less feature than the original one
    if not len(test_data) == 0 and not len(test_data[0]) == 0:
        feature_count = len(data[0]) # how many feature available in the data
        for i in xrange(len(test_data)):
            new_record = []
            for j in xrange(feature_count):
                if gene[j] == '1':
                    new_record.append(test_data[i][j])
            new_test_data.append(new_record)
    new_svc =  ga.best_individuals(1,representation='svc', benchmark=used_benchmark)
    
    # the casual svm
    casual_svc = svm.SVC(kernel='linear').fit(training_data, training_target)
    
    # calculate prediction
    new_svc_prediction = new_svc.predict(new_test_data)
    casual_svc_prediction = casual_svc.predict(test_data)
    
    # compare error
    new_svc_error_count = 0
    casual_svc_error_count = 0
    new_svc_error_square_sum = 0
    casual_svc_error_square_sum = 0
    for i in xrange(len(test_data)):
        if new_svc_prediction[i] <> test_target[i]:
            new_svc_error_count = new_svc_error_count + 1
        if casual_svc_prediction[i] <> test_target[i]:
            casual_svc_error_count = casual_svc_error_count + 1
        new_svc_error_square_sum += (test_target[i] - new_svc_prediction[i])**2
        casual_svc_error_square_sum += (test_target[i] - casual_svc_prediction[i])**2
    new_svc_mse = new_svc_error_square_sum/len(test_data)
    casual_svc_mse = casual_svc_error_square_sum/len(test_data)
    
    # dataset
    print('='*20+' dataset '+'='*20)
    for i in xrange(len(test_data)):
        print('actual_features : %s, new_svc_features : %s' %(test_data[i], new_test_data[i]))
    # prediction
    print('='*20+' prediction '+'='*20)
    for i in xrange(len(test_data)):
        print('target : %f, new_svc_prediction : %f, casual_svc_prediction : %f' %(test_target[i], new_svc_prediction[i], casual_svc_prediction[i]))
    # conclusion
    print('='*20+' conclusion '+'='*20)
    print('new_svc_error : %d, casual_svc_error : %d' %(new_svc_error_count, casual_svc_error_count))
    print('new_svc_mse : %f, casual_svc_mse : %f' %(new_svc_mse, casual_svc_mse))
    print('new_svc_feature_count : %d, casual_svc_feature_count : %d' %(len(new_test_data[0]), len(test_data[0])))
    
    # show genetics algorithm graphic
    ga.show()
    
