#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit    
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
DATA_SET = "../final_project_dataset.pkl"


feature_order = {'salary':True,
 'to_messages':True,
 'deferral_payments':True,
 'total_payments':True,
 'exercised_stock_options':True,
 'bonus':True,
 'restricted_stock':True,
 'shared_receipt_with_poi':True,
 'restricted_stock_deferred':False,
 'total_stock_value':True,
 'expenses':True,
 'loan_advances':False,
 'from_messages':True,
 'other':False,
 'from_this_person_to_poi':True,
 'director_fees':True,
 'deferred_income':False,
 'long_term_incentive':True,
 'from_poi_to_this_person':True,
 'nan_number':False,
 'sort_score':True}

#Following is some functions calculating various number is the analysis    
def median(lst):
    return np.median(np.array(lst))

def find_position(dataset, feature_name, identity_name, reverse):
    features_all = [dataset[key][feature_name] for key in dataset]
    features_all = [key for key in features_all if key != 'NaN']
    features_all = list(features_all)
    features_all.sort(reverse=reverse)
    value = dataset[identity_name][feature_name]
    if value == 'NaN':
        position = len(features_all)/2.0;
    else:
        position = features_all.index(value)+1.
    return position

def position_to_score(position, power):  
    score = 1./(position**power)
    return score

def calculate_sort_score(dataset, identity_name, features_names, power, features_weights):
    total_score = 0
    for feature in features_names:
        if feature in features_weights:
            total_score = total_score + features_weights[feature]*position_to_score(find_position(dataset, feature, identity_name, feature_order[feature]), power)
        else:
            total_score = total_score + position_to_score(find_position(dataset, feature, identity_name, feature_order[feature]), power)
    return total_score

def threshold_classifier(single_features, threshold):
    result = []
    for feature in single_features:
        if feature > threshold:
            result.append(1)
        else:
            result.append(0)
    return result

def calculate_accumutive_integral(dataset, feature_name, order):
    feature_list = [(dataset[key][feature_name], dataset[key]['poi']) for key in dataset]
    feature_median = median([value[0] for value in feature_list if value[0] != 'NaN'])
    feature_list = [(feature_median, value[1]) if value[0] == 'NaN' else value for value in feature_list]
    feature_list = sorted(feature_list, key=lambda value: value[0], reverse = order)
    accu_numbers = list()
    accu_number_pois = 0
    integral = 0
    for index in range(len(feature_list)):
        if feature_list[index][1] == True:
            accu_number_pois = accu_number_pois + 1
        accu_numbers.append(accu_number_pois)
        integral = integral + accu_number_pois
    return list(range(len(feature_list))), accu_numbers, integral

def replace_nan_median(dataset, feature_name):
    numbers_list = [dataset[key][feature_name] for key in dataset if dataset[key][feature_name] != 'NaN']
    number_median = median(numbers_list)
    for key in dataset.keys():
        if dataset[key][feature_name] == 'NaN':
            dataset[key][feature_name] = number_median
    return dataset
    
def replace_nan_zero(dataset, feature_name):
    for key in dataset.keys():
        if dataset[key][feature_name] == 'NaN':
            dataset[key][feature_name] = 0
    return dataset
def replace_nan_average_rank(dataset, features_list, feature_order):
    average_rank = {}    
    instance_number = len(dataset)    
    for person in dataset:
        temp_list = []
        for feature in features_list:
            if dataset[person][feature] != 'NaN':
                temp_list.append(find_position(dataset, feature, person, reverse=feature_order[feature]))
        average_rank[person] = int(round(np.mean(temp_list)))
    for feature in features_list:
        print feature
        numbers_list = [dataset[key][feature] for key in dataset if dataset[key][feature] != 'NaN']
        all_list = [number for number in numbers_list]    
        non_nan_count = len(numbers_list)
        nan_count = instance_number-non_nan_count
        print nan_count
        if nan_count > 0:
            for i in range(nan_count):
                all_list.extend(random.sample(numbers_list, 1))
            all_list.sort(reverse = feature_order[feature])
            for person in dataset:
                if dataset[person][feature] == 'NaN':
                    dataset[person][feature] = all_list[average_rank[person]-1]
    return dataset
    
def count_nan(a_data_dict):
    return len([0 for key in a_data_dict if a_data_dict[key] == 'NaN'])

def dict_to_np_array(dict_dataset, extracted_features):
    data_array = []
    names_list = []
    for key in dict_dataset:
        names_list.append(key)
        temp = []
        for feature in extracted_features: 
            temp.append(dict_dataset[key][feature])
        data_array.append(np.array(temp))
    return np.array(data_array), names_list

def incorp_array_dict(dict_dataset, array_dataset, names_list, features_list):
    for index in range(len(names_list)):
        for index0 in range(len(features_list)):
            dict_dataset[names_list[index]][features_list[index0]] = array_dataset[index][index0]
    return dict_dataset

def calculate_scaled_features(dict_dataset, features_list):
    data_array, names_list = dict_to_np_array(dict_dataset, features_list)
    data_array_scaled = preprocessing.scale(data_array)
    dict_dataset = incorp_array_dict(dict_dataset, data_array_scaled, names_list, features_list)
    return dict_dataset

def test_classifier_values(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        return precision, recall, f1
    except:
        return 0,0,0

def test_classifier_values_tuning(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        print clf.best_params_
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        return precision, recall, f1
    except:
        return 0,0,0
    
def plot_classify_hist(my_dataset, feature_name):
    pois = [my_dataset[key][feature_name] for key in my_dataset if my_dataset[key]['poi'] == True]
    non_pois = [my_dataset[key][feature_name] for key in my_dataset if my_dataset[key]['poi'] == False]
    num_bins = 50
    n, bins, patches = plt.hist(pois, num_bins, normed=1, facecolor='green', alpha=0.5)
    n, bins, patches = plt.hist(non_pois, num_bins, normed=1, facecolor='red', alpha=0.5)
    plt.show()

def find_threshold(dataset, feature_name):
    result = None
    features_data = [(dataset[key][feature_name], dataset[key]['poi']) for key in dataset]
    features_data.sort(key = lambda value: value[0], reverse = not feature_order[feature_name])
    for i in range(len(features_data)):
        if features_data[i][1] == False and features_data[i+1][1] == True:
            for j in range(i, -1, -1):
                if features_data[j][0] != features_data[i][0]:
                    result = features_data[j][0]
                    break
        if result != None:
            break
    return result

def delete_threshold(dataset, feature_name, threshold):
    if feature_order[feature_name]:
        result = {key:dataset[key] for key in dataset if dataset[key][feature_name] > threshold}
    else:
        result = {key:dataset[key] for key in dataset if dataset[key][feature_name] < threshold}
    return result

def filter_data(dataset, feature_name):
    return delete_threshold(dataset, feature_name, find_threshold(dataset, feature_name))

def add_poi(dataset, poi_number):
    random.seed(1)
    poi_data = [(key, dataset[key]) for key in dataset if dataset[key]['poi'] == True]
    for i in range(poi_number):
        random_index = random.choice(range(len(poi_data)))
        dataset[poi_data[random_index][0] + str(i)] = poi_data[random_index][1]
    return dataset

print 'Loading data...'
with open(DATA_SET, "r") as data_file:
    data_dict = pickle.load(data_file)
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
print 'Loading complete'
print '-----------------------------'
all_feature_names = my_dataset[my_dataset.keys()[1]].keys()
poi_number = len([key for key in my_dataset if my_dataset[key]['poi'] == True])
non_poi_number = len([key for key in my_dataset if my_dataset[key]['poi'] == False])
number_features_names = [name for name in all_feature_names]
number_features_names.remove('email_address')
number_features_names.remove('poi')

print 'Identify outliers by following plot...'
plot_data = [(my_dataset[key]['total_payments'], my_dataset[key]['total_stock_value']) for key in my_dataset]
total_payments = [n[0] for n in plot_data]
total_stock_value = [n[1] for n in plot_data]
total_payments = [x if x != 'NaN' else 0 for x in total_payments]
total_stock_value = [x if x != 'NaN' else 0 for x in total_stock_value]
name_maxpayment = [key for key in my_dataset if my_dataset[key]['total_payments'] == sorted(total_payments, reverse = True)[0]][0]
name_2ndpayment = [key for key in my_dataset if my_dataset[key]['total_payments'] == sorted(total_payments, reverse = True)[1]][0]
plt.scatter(total_payments, total_stock_value)
plt.xlabel('Total Payments')
plt.ylabel('Total Stock Value')
plt.annotate('outlier: ' + '"' + name_maxpayment + '"', \
            xy=(my_dataset[name_maxpayment]['total_payments'], my_dataset[name_maxpayment]['total_stock_value']), \
            xytext=(my_dataset[name_maxpayment]['total_payments'] - 1.5e8, my_dataset[name_maxpayment]['total_stock_value']),\
            arrowprops=dict(facecolor='red', shrink=0.05))
plt.annotate('outlier: ' + '"' + name_2ndpayment + '"', \
            xy=(my_dataset[name_2ndpayment]['total_payments'], my_dataset[name_2ndpayment]['total_stock_value']), \
            xytext=(my_dataset[name_2ndpayment]['total_payments'] + 0.6e8, my_dataset[name_2ndpayment]['total_stock_value']),\
            arrowprops=dict(facecolor='red', shrink=0.05))
plt.savefig('figures/payments_verse_stock.png', dpi=300)
plt.show()
print 'Outliers processed...'
print '-----------------------------'
my_dataset.pop(name_maxpayment)
all_persons = my_dataset.keys()

print 'A statistical test for NaN numbers...'
poi_nan = []
nonpoi_nan = []
for person in all_persons:
    if my_dataset[person]['poi'] == True:
        poi_nan.append(count_nan(my_dataset[person]))
    else:
        nonpoi_nan.append(count_nan(my_dataset[person]))
mean_poi = np.mean(poi_nan)
mean_nonpoi = np.mean(nonpoi_nan)
std = np.sqrt(np.var(poi_nan)/len(poi_nan) + np.var(nonpoi_nan)/len(nonpoi_nan))
t_statistics = (mean_poi - mean_nonpoi)/std
p_value = stats.t.cdf(t_statistics, len(poi_nan) + len(nonpoi_nan) - 2)
print 't statistical is ', t_statistics,'. ', 'And p value is ', p_value, '. '
print '-----------------------------'

print 'First new feature which is NaN number is created....'
for person in all_persons:
    my_dataset[person]['nan_number'] = count_nan(my_dataset[person])
number_features_names.append('nan_number')
print '-----------------------------'

print 'Dealing with NaNs....(replacing with medians)'
for feature in number_features_names:
    my_dataset = replace_nan_median(my_dataset, feature)
    #my_dataset = replace_nan_zero(my_dataset, feature)
#my_dataset = replace_nan_average_rank(my_dataset, number_features_names, feature_order)
print '-----------------------------'

print 'Second new feature which is NaN number will be created...'
print '1. Computing the sorted integrals of every feature and choose first 6 to construct sort score.'
print 'Plot of integral of total stock value is as following:'
total_stock_poi_integral = calculate_accumutive_integral(my_dataset, 'total_stock_value', feature_order['total_stock_value'])
fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
big_ax = fig.add_subplot(111)
big_ax.set_axis_bgcolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
min_total = min(total_stock_poi_integral[1])
ax1.plot(total_stock_poi_integral[0], total_stock_poi_integral[1], lw=2)
ax2.fill_between(total_stock_poi_integral[0], min_total, total_stock_poi_integral[1], facecolor='blue', alpha=0.5)
for ax in ax1, ax2:
    ax.grid(True)
ax1.set_ylabel('Accumutive Number of POIs')
for label in ax2.get_yticklabels():
    label.set_visible(False)
fig.suptitle('Accumutive POIs as Total Stock Value Reduced')
big_ax.set_xlabel('Sorted Position Numbers')
fig.savefig('figures/total_stock_integral.png', dpi=300)
plt.show()

print 'Plot of integral of director fees is as following:'
from_message_poi_integral = calculate_accumutive_integral(my_dataset, 'director_fees', feature_order['director_fees'])
fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
big_ax = fig.add_subplot(111)
big_ax.set_axis_bgcolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
min_total = min(from_message_poi_integral[1])
ax1.plot(from_message_poi_integral[0], from_message_poi_integral[1], lw=2)
ax2.fill_between(from_message_poi_integral[0], min_total, from_message_poi_integral[1], facecolor='blue', alpha=0.5)
for ax in ax1, ax2:
    ax.grid(True)
ax1.set_ylabel('Accumutive Number of POIs')
for label in ax2.get_yticklabels():
    label.set_visible(False)
fig.suptitle('Accumutive POIs as Number of dierctor fees Reduced')
big_ax.set_xlabel('Sorted Position Numbers')
fig.savefig('figures/director_fees_integral.png', dpi=300)
plt.show()

integrals = [(feature, calculate_accumutive_integral(my_dataset, feature, feature_order[feature])[2]) for feature in number_features_names]
integrals.sort(key = lambda value: value[1], reverse = True)
features_names = [value[0] for value in integrals][0:6]
with open("results/integrals_sort.txt", "w+") as integral_file:
    for data in integrals:
        integral_file.write(data[0])
        integral_file.write(', ')
        integral_file.write(str(data[1]))
        integral_file.write('\n')
features_weights = {'total_stock_value':1}
print 'Sorted integrals is as following:'
print integrals
print 'Features used to create sorted scores are:'
print features_names
print '-----------------------------'

print '2. Choosing right p parameter....'
powers_list = [0.2,0.4, 1, 2, 3, 5, 10]
power_scores=[]
for power in powers_list:
    for person in all_persons:
        my_dataset[person]['sort_score'] = calculate_sort_score(my_dataset, person, features_names, power,features_weights) 
    features_list = ['poi', 'sort_score']       
    clf = GaussianNB()
    score = test_classifier_values(clf, my_dataset, features_list, folds = 1000)  
    power_scores.append(tuple([power, score[0], score[1], score[2]]))
power_scores.sort(key = lambda value: value[3], reverse = True)
with open("results/p_parameter_sort.txt", "w+") as p_parameter_file:
    for data in power_scores:
        p_parameter_file.write(str(data[0]))
        p_parameter_file.write(', ')
        p_parameter_file.write(str(data[1]))
        p_parameter_file.write(', ')
        p_parameter_file.write(str(data[2]))
        p_parameter_file.write(', ')
        p_parameter_file.write(str(data[3]))
        p_parameter_file.write('\n')
features_weights = {'total_stock_value':1}
print 'Sorted performance of different p is as following:'
print power_scores
print 'p paramter which has best porfermance is ', power_scores[0][0], '. '
print '-----------------------------'
print '3. Second new feature which is sort score is created....'
power = power_scores[0][0]
for person in all_persons:
    my_dataset[person]['sort_score'] = calculate_sort_score(my_dataset, person, features_names, power, features_weights) 
number_features_names.append('sort_score')
print '-----------------------------'

print 'Feature scaling by standardizing...'
my_dataset = calculate_scaled_features(my_dataset, number_features_names)
print '-----------------------------'

print 'Solving unbalance classes by oversampling...'
my_dataset = add_poi(my_dataset, non_poi_number - poi_number)
print '-----------------------------'

print 'Selecting features by sorting scores...'
feature_scores=[]
for feature in number_features_names:
    features_list = ['poi', feature] 
    clf = GaussianNB()
    score = test_classifier_values(clf, my_dataset, features_list, folds = 1000)
    feature_scores.append(tuple([feature, score[0], score[1], score[2]]))
print 'Performance of different features is as following(precison, recall, f1):'
print feature_scores
print '-----------------------------'

top_numbers = 20
feature_scores.sort(key = lambda value: value[1], reverse = True)
precision_tops = [value[0] for value in feature_scores][0:top_numbers]
feature_scores.sort(key = lambda value: value[2], reverse = True)
recall_tops = [value[0] for value in feature_scores][0:top_numbers]
feature_scores.sort(key = lambda value: value[3], reverse = True)
f1_tops = [value[0] for value in feature_scores][0:top_numbers]


print 'Selecting number of features by sorting scores...'
n_scores = []
for selected_number in range(1,top_numbers + 1):
    features_list = set()
    clf = SVC()
    for index in range(selected_number):
        features_list.add(f1_tops[index])
        features_list.add(precision_tops[index])
    features_list = list(features_list)
    features_list.insert(0, 'poi')
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    clf.fit(features_train, labels_train)
    score = test_classifier_values(clf, my_dataset, features_list, folds = 1000)
    n_scores.append(tuple([selected_number, score[0], score[1], score[2]]))
print 'Performance of different n is as following(precision, recall, f1):'
print n_scores


with open("results/n_scores.txt", "w+") as n_scores_file:
    for data in n_scores:
        n_scores_file.write(str(data[0]))
        n_scores_file.write(', ')
        n_scores_file.write(str(data[1]))
        n_scores_file.write(', ')
        n_scores_file.write(str(data[2]))
        n_scores_file.write(', ')
        n_scores_file.write(str(data[3]))
        n_scores_file.write('\n')

selected_number = 8
features_list = set()
for index in range(selected_number):
    features_list.add(f1_tops[index])
    features_list.add(precision_tops[index])
features_list = list(features_list)
print 'Each of ', selected_number, ' tops features of precision and f1 score are selected.'
print 'They are following:'
print features_list
features_list.insert(0, 'poi')
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
print '-----------------------------'



algorithm_scores_file  = open("results/algorithm_scores.txt", "w+")
print 'Three algorithm will be compared.'
print 'Frist is Gaussian Naive Bayes.'
clf = GaussianNB()
clf.fit(features_train, labels_train)
score = test_classifier_values(clf, my_dataset, features_list, folds = 1000)
print 'Precison is ', score[0], '.'
print 'Recal is ', score[1], '.'
print 'F1 is ', score[2], '.'
algorithm_scores_file.write('Gaussian Naive Bayes')
algorithm_scores_file.write(', ')
algorithm_scores_file.write(str(score[0]))
algorithm_scores_file.write(', ')
algorithm_scores_file.write(str(score[1]))
algorithm_scores_file.write(', ')
algorithm_scores_file.write(str(score[2]))
algorithm_scores_file.write('\n')
print '-----------------------------'

print 'Second is Decison Tree.'
clf = tree.DecisionTreeClassifier(min_samples_leaf = 2)
clf.fit(features_train, labels_train)
score = test_classifier_values(clf, my_dataset, features_list, folds = 1000)
print 'Precison is ', score[0], '.'
print 'Recal is ', score[1], '.'
print 'F1 is ', score[2], '.'
algorithm_scores_file.write('Decision Tree')
algorithm_scores_file.write(', ')
algorithm_scores_file.write(str(score[0]))
algorithm_scores_file.write(', ')
algorithm_scores_file.write(str(score[1]))
algorithm_scores_file.write(', ')
algorithm_scores_file.write(str(score[2]))
algorithm_scores_file.write('\n')
print '-----------------------------'


print 'Third is tuned Support Vector Machine.'
parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'),\
              'C':[0.1, 1, 10,100,1000,10000,30000]}
svr = SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(features_train, labels_train)
print 'Parameter kernel and C is tuned.'
with open("results/tuning_scores.txt", "w+") as tuning_scores_file:
    for data in clf.grid_scores_:
        tuning_scores_file.write(str(data[0]['C']))
        tuning_scores_file.write(', ')
        tuning_scores_file.write(data[0]['kernel'])
        tuning_scores_file.write(', ')
        tuning_scores_file.write(str(data[1]))
        tuning_scores_file.write('\n')
print 'Scores of different parameters are following:'
print clf.grid_scores_
print 'Best C is ', clf.best_params_['C'], '.'
print 'Best kernel is ', clf.best_params_['kernel'], '.'
clf = SVC(C = clf.best_params_['C'], kernel = clf.best_params_['kernel'])
clf.fit(features_train, labels_train)
score = test_classifier_values(clf, my_dataset, features_list, folds = 1000)
print 'Precison is ', score[0], '.'
print 'Recal is ', score[1], '.'
print 'F1 is ', score[2], '.'
algorithm_scores_file.write('Tuned Support Vector Machine')
algorithm_scores_file.write(', ')
algorithm_scores_file.write(str(score[0]))
algorithm_scores_file.write(', ')
algorithm_scores_file.write(str(score[1]))
algorithm_scores_file.write(', ')
algorithm_scores_file.write(str(score[2]))
algorithm_scores_file.write('\n')
algorithm_scores_file.close()
print '-----------------------------'



dump_classifier_and_data(clf, my_dataset, features_list)

