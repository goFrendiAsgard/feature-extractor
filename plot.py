from Feature_Extractor import *
import pylab as pl
import numpy as np
from sklearn.svm import SVC

# parameters

# synthesis 03
records                     = extract_csv('synthesis_03.csv')
ommited_class_for_plotting  = ['D','E']
new_features                = [
                               '(f2) / (f1)',
                               '(f1) / (f3)'
                              ]


'''
# ilustration-ori
records                     = extract_csv('segitiga.csv')
ommited_class_for_plotting  = []
new_features                = [
                               'x',
                               'y'
                              ]
'''

'''
# ilustration-extracted
records                     = extract_csv('segitiga.csv')
ommited_class_for_plotting  = []
new_features                = [
                               'sqr(x)+sqr(y)',
                               '0'
                              ]
'''


'''
# ideal
records                     = extract_csv('synthesis_01.csv')
ommited_class_for_plotting  = []
new_features                = [
                               '(abs(attack)) / (defense)',
                               '(exp(stamina)) - (exp(agility))'
                              ]
'''


projection_feature_indexes = [0,1]
records = [item for item in records if item[-1] not in ommited_class_for_plotting]
groups = []
for record in records[1:]:
    if record[-1] not in groups:
        groups.append(record[-1])
group_count = len(groups)
clf = SVC()
#clf = DecisionTreeClassifier(max_depth=group_count-1, random_state=0)
# calculate new_data, label_targets and numeric_targets
new_data = []
old_data = []
label_targets = []
numeric_targets = []
old_features = records[0][:-1]
target_dict = {}
index = 1
for record in records[1:]:
    old_data.append(record[:-1])
    label_target = record[-1]
    label_targets.append(label_target)
    if not label_target in target_dict:
        target_dict[label_target] = index
        index += 1
    numeric_targets.append(target_dict[label_target])
for i in xrange(len(old_data)):
    new_data.append([])
for new_feature in new_features:
    projection = get_projection(new_feature, old_features, old_data)
    for i in xrange(len(old_data)):
        new_data[i].append(projection[i])
# train classifier
clf.fit(new_data, numeric_targets)
# get minimum and maximum x & y
new_data = np.array(new_data)

x_min, x_max = new_data[:, projection_feature_indexes[0]].min() - 0.05, new_data[:, projection_feature_indexes[0]].max() + 0.05
y_min, y_max = new_data[:, projection_feature_indexes[1]].min() - 0.05, new_data[:, projection_feature_indexes[1]].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001),
                     np.arange(y_min, y_max, 0.001))
    
# plotting
pl.subplot(1,1,1)
if len(new_features)>2:
    tup = xx.ravel(), yy.ravel()
    for i in xrange(len(new_features)-2):
        tup = tup + ( [0.5]*len(xx.ravel()) , )
    Z = clf.predict(np.c_[tup])
else:
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
pl.xlabel(new_features[projection_feature_indexes[0]])
pl.ylabel(new_features[projection_feature_indexes[1]])
pl.contourf(xx, yy, Z, cmap = pl.cm.RdBu, alpha=.5)

# Plot also the training points
p_dict = {}
x_dict = {}
y_dict = {}
z_dict = {}
for label_target in target_dict:
    p_dict[label_target] = []
    x_dict[label_target] = []
    y_dict[label_target] = []
    z_dict[label_target] = []

for element_index in xrange(len(new_data)):
    x = new_data[element_index, projection_feature_indexes[0]]
    y = new_data[element_index, projection_feature_indexes[1]]
    z = numeric_targets[element_index]
    for label_target in target_dict:
        if z==target_dict[label_target]:
            x_dict[label_target].append(x)
            y_dict[label_target].append(y)
            z_dict[label_target].append(z)
            break
i=3
color_list = ['white', 'blue', 'red', 'purple', 'green']
for label_target in target_dict:
    p_dict[label_target] = pl.scatter(x_dict[label_target], y_dict[label_target], s=90, c=color_list[(i-3)%5], marker=(i,0))
    i+=1

p_list = []
label_list = []
for label_target in target_dict:
    p_list.append(p_dict[label_target])
    label_list.append(label_target)

#pl.scatter(new_data[:, projection_feature_indexes[0]], new_data[:, projection_feature_indexes[1]], s=32, c=numeric_targets, cmap=plt.cm.gist_rainbow)
#pl.scatter(new_data[:, projection_feature_indexes[0]], new_data[:, projection_feature_indexes[1]], s=32, c=numeric_targets, cmap=plt.cm.gist_rainbow)

pl.title('plot')
pl.legend(p_list, label_list)
print label_list
print p_list

prediction = clf.predict(new_data)
correct = 0.0
for i in xrange(len(prediction)):
    if prediction[i] == numeric_targets[i]:
        correct += 1
print correct, len(prediction), correct/len(prediction)


pl.show()

