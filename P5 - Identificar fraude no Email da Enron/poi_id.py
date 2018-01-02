# encoding: utf-8
# encoding: iso-8859-1
# encoding: win-1252

import sys
import pickle
from collections import defaultdict

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
import pprint
import warnings

# Desativa mensagens de warning
warnings.filterwarnings('ignore')

'''
Abaixo estão os recursos que o nosso conjunto de dados contém. 
Mas não iremos usar todos os recursos fornecidos para análises futuras.

# Unidade está em dolares
financial_features = ['salary',
                      'deferral_payments',
                      'total_payments',
                      'loan_advances',
                      'bonus',
                      'restricted_stock_deferred',
                      'deferred_income',
                      'total_stock_value',
                      'expenses',
                      'exercised_stock_options', 
                      'other', 
                      'long_term_incentive', 
                      'restricted_stock',
                      'director_fees']

email_features = ['to_messages',
                  'email_address', 
                  'from_poi_to_this_person', 
                  'from_messages',
                  'from_this_person_to_poi', 
                  'shared_receipt_with_poi']

# Valor booleano represetando por inteiro
POI_label= ['poi'] 
'''

initial_features_list = ['poi',
                         'salary',
                         'deferral_payments',
                         'total_payments',
                         'loan_advances',
                         'bonus',
                         'restricted_stock_deferred',
                         'deferred_income',
                         'total_stock_value',
                         'expenses',
                         'exercised_stock_options',
                         'other',
                         'long_term_incentive',
                         'restricted_stock',
                         'director_fees',
                         'to_messages',
                         'from_poi_to_this_person',
                         'from_messages',
                         'from_this_person_to_poi',
                         'shared_receipt_with_poi']

### Carrega o dataset num dict
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Declarando variáveis
number_of_poi = 0
number_of_non_poi = 0
null_count = defaultdict(int)

for name, features in data_dict.iteritems():
    if features['poi']:
        number_of_poi += 1
    else:
        number_of_non_poi += 1
    for key, value in features.iteritems():
        if value == 'NaN':
            null_count[key] += 1

# Como sabemos o nome do presidente da Enron, podemos usar o nome dele como referencia para pegar a quantidade de features
number_of_features = len(data_dict['LAY KENNETH L'])
print "================================Informações básicas==============================="
print("\nNúmero total de pessoas no conjunto de dados: {}".format(len(data_dict)))
print("\nNúmero total de pessoas de interesse (POI) no conjunto de dados: {}".format(number_of_poi))
print("\nNúmero total de pessoas de não interesse (non POI) no conjunto de dados: {}".format(number_of_non_poi))
print("\nCada pessoa possui {} features e está listada abaixo:".format(number_of_features))

for key in data_dict['LAY KENNETH L'].keys():
    print key

print "\nNúmero total de features sem valores:"
for key in null_count.keys():
    print(key, null_count[key])

###  Remove outliers
'''
Vamos dar uma olhada nos dados para verificar se existe outliers. 
Selecionei dois recursos que são "salário" e "bônus".
Vou traçar os dois para que possamos encontrar se há alguns valores abertos.
'''
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### Plotando os features
import matplotlib.pyplot as plt

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")

print "\n==============================Verificando os outliers============================="
plt.show()

# Verificando manualmente os dados, encontrei 'TOTAL' e 'THE TRAVEL AGENCY IN THE PARK' que irei removê-los
data_dict.pop('TOTAL', None)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)  # it was no way related to our motto so I removed
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")

print "\n===========================Depois de remover os outliers=========================="
print "Foi removido 'TOTAL' e 'THE TRAVEL AGENCY IN THE PARK'"
plt.show()

### Verificando outros outliers
outliers = []
for key in data_dict:
    value = data_dict[key]['salary']
    if value == 'NaN':
        continue
    outliers.append((key, int(value)))

Top_outliers = (sorted(outliers, key=lambda x: x[1], reverse=True)[:4])

# Como estes outliers são do nosso interesse, não irei remover
print "\nTop 4 outliers:"
print Top_outliers

### Exportando facilmente a variável, atribuindo numa nova variável
initial_dataset = data_dict

### Extraindo features e labels do conjunto de dados para testes locais
data = featureFormat(initial_dataset, initial_features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Escalando as features
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
scaled_features = min_max_scaler.fit_transform(features)

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(scaled_features, labels, test_size=0.3, random_state=42)

# Experimentar variedades de classificadores
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from time import time

# Definindo uma função que receberá um tipo de classificador e mostrará score de accuracy, precision e recall
def classify(clf):
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print "Accuracy :", accuracy_score(pred, labels_test)
    print "Precision Score :", precision_score(pred, labels_test)
    print "Recall Score :", recall_score(pred, labels_test)

print "\n=======================Antes de adicionar as novas features======================="
print "\nVerificando GaussianNB :"
classify(GaussianNB())

print "\nVerificando DecisionTreeClassifier"
classify(DecisionTreeClassifier())

print "\nVerificando SVC"
classify(SVC())

print "\nVerificando RandomForestClassifier"
classify(RandomForestClassifier())

###  Criação de novas features
for name, fetaures in data_dict.items():
    # Porcentagem dessa pessoa para POI
    if fetaures['from_messages'] == 'NaN' or fetaures['from_this_person_to_poi'] == 'NaN':
        fetaures['fraction_from_this_person_to_poi'] = 0.0
    else:
        fetaures['fraction_from_this_person_to_poi'] = fetaures['from_this_person_to_poi'] / float(fetaures['from_messages'])

    # Porcentagem do POI para esta pessoa
    if fetaures['to_messages'] == 'NaN' or fetaures['from_poi_to_this_person'] == 'NaN':
        fetaures['fraction_from_poi_to_this_person'] = 0.0
    else:
        fetaures['fraction_from_poi_to_this_person'] = fetaures['from_poi_to_this_person'] / float(fetaures['to_messages'])

# Agora executar o mesmo procedimento anteriro agora com os novos features
my_dataset = data_dict

# Adicionando os novos features na lista 'initial_features_list'
initial_features_list.extend(['fraction_from_poi_to_this_person', 'fraction_from_this_person_to_poi'])

data_ = featureFormat(my_dataset, initial_features_list, sort_keys=True)
labels, features = targetFeatureSplit(data_)

# Escalando os features
min_max_scaler = preprocessing.MinMaxScaler()
scaled_features = min_max_scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test = train_test_split(scaled_features, labels, test_size=0.3, random_state=42)

# Vamos ver como o desempenho é afetado após a adição de novos features
print "\n=========================Após inclusão dos novos features========================="
print "Features novos: fraction_from_poi_to_this_person e fraction_from_this_person_to_poi"
print "\nVerificando GaussianNB :"
classify(GaussianNB())

print "\nVerificando DecisionTreeClassifier"
classify(DecisionTreeClassifier())

print "\nVerificando SVC"
classify(SVC())

print "\nVerificando RandomForestClassifier"
classify(RandomForestClassifier())


# Função que calculca score de precision, recall e f1
def score_func(y_true, y_predict):
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for prediction, truth in zip(y_predict, y_true):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        else:
            true_positives += 1
    if true_positives == 0:
        return (0, 0, 0)
    else:
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        return (precision, recall, f1)


# Função que separa cada feature com valores do score de precision, recall e f1
def univariateFeatureSelection(f_list, my_dataset):
    result = []
    for feature in f_list:
        # Replace 'NaN' with 0
        for name in my_dataset:
            data_point = my_dataset[name]
            if not data_point[feature]:
                data_point[feature] = 0
            elif data_point[feature] == 'NaN':
                data_point[feature] = 0

        data = featureFormat(my_dataset, ['poi', feature], sort_keys=True, remove_all_zeroes=False)
        labels, features = targetFeatureSplit(data)
        features = [abs(x) for x in features]
        from sklearn.cross_validation import StratifiedShuffleSplit
        cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for train_idx, test_idx in cv:
            for ii in train_idx:
                features_train.append(features[ii])
                labels_train.append(labels[ii])
            for jj in test_idx:
                features_test.append(features[jj])
                labels_test.append(labels[jj])
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        score = score_func(labels_test, predictions)
        result.append((feature, score[0], score[1], score[2]))
    result = sorted(result, reverse=True, key=lambda x: x[3])
    return result


# Pegando os valores do features
univariate_result = univariateFeatureSelection(initial_features_list, my_dataset)

print "\n=================================================================================="
print '\nFeatures com score de precision, recall e f1 respectivamente:'
for l in univariate_result:
    print l

# Criaçao de novo filtro de features
features_list = ['poi',
                 'total_stock_value',
                 'exercised_stock_options',
                 'bonus',
                 'deferred_income',
                 'long_term_incentive',
                 'restricted_stock',
                 'salary',
                 'total_payments',
                 'other',
                 'shared_receipt_with_poi']

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

min_max_scaler = preprocessing.MinMaxScaler()
scaled_features = min_max_scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test = train_test_split(scaled_features, labels, test_size=0.3, random_state=42)

print "\n=================================================================================="
print '''\nExaminado agora com as seguintes features:
* poi
* total_stock_value
* exercised_stock_options
* bonus
* deferred_income
* long_term_incentive
* restricted_stock
* salary
* total_payments
* other
* shared_receipt_with_poi 
'''

print "\nVerificando GaussianNB :"
classify(GaussianNB())

print "\nVerificando DecisionTreeClassifier"
classify(DecisionTreeClassifier())

print "\nVerificando SVC"
classify(SVC())

print "\nVerificando RandomForestClassifier"
classify(RandomForestClassifier())

###  Ajustando o classificador para obter melhores precisões e recall com tamanho 0.3
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

print"\n==============GaussianNB============="

gauss = GaussianNB()
nb_pipe = Pipeline([('scaler', MinMaxScaler()), ('selection', SelectKBest()), ('pca', PCA()), ('naive_bayes', gauss)])

nb_parameters = [{'selection__k': [8, 9, 10], 'pca__n_components': [6, 7, 8]}]

nb_grid = GridSearchCV(estimator=nb_pipe,
                       param_grid=nb_parameters,
                       n_jobs=-1,
                       cv=StratifiedKFold(labels_train, n_folds=6, shuffle=True),
                       scoring='f1')

start_fitting = time()
nb_grid.fit(features_train, labels_train)
end_fitting = time()
print("Training time : {}".format(end_fitting - start_fitting))

start_predicting = time()
nb_pred = nb_grid.predict(features_test)
end_predicting = time()
print("Predicting time : {}".format(end_predicting - start_predicting))

nb_accuracy = accuracy_score(nb_pred, labels_test)
print('Naive Bayes accuracy : {}'.format(nb_accuracy))
print "f1 score :", f1_score(nb_pred, labels_test)
print "precision score :", precision_score(nb_pred, labels_test)
print "recall score :", recall_score(nb_pred, labels_test)
print(nb_grid.best_estimator_)

print "\n==============DecisionTreeClassifier==============="
decision_tree = DecisionTreeClassifier()

dt_params = [{'min_samples_split': [2, 3, 4], 'criterion': ['gini', 'entropy']}]

dt_grid = GridSearchCV(estimator=decision_tree,
                       param_grid=dt_params,
                       cv=StratifiedKFold(labels_train, n_folds=6, shuffle=True),
                       n_jobs=-1,
                       scoring='f1')

start_fitting = time()
dt_grid.fit(features_train, labels_train)
end_fitting = time()
print("Training time : {}".format(round(end_fitting - start_fitting, 3)))

start_predicting = time()
dt_pred = dt_grid.predict(features_test)
end_predicting = time()
print("Predicting time : {}".format(round(end_predicting - start_predicting, 3)))

dt_accuracy = accuracy_score(dt_pred, labels_test)
print('Decision Tree accuracy : {}'.format(dt_accuracy))
print "f1 score :", f1_score(dt_pred, labels_test)
print "precision score :", precision_score(dt_pred, labels_test)
print "recall score :", recall_score(dt_pred, labels_test)
print(dt_grid.best_estimator_)

print "\n=================SVC================="

svc_pipe = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC())])

svc_params = {'svc__kernel': ['linear', 'rbf'],
              'svc__C': [0.1, 1, 10, 100, 1000],
              'svc__gamma': [1e-3, 1e-4, 1e-1, 1, 10]}

svc_grid = GridSearchCV(estimator=svc_pipe,
                        param_grid=svc_params,
                        cv=StratifiedKFold(labels_train, n_folds=6, shuffle=True),
                        n_jobs=-1,
                        scoring='f1')

start_fitting = time()
svc_grid.fit(features_train, labels_train)
end_fitting = time()
print("Training time : {}".format(end_fitting - start_fitting))

start_predicting = time()
svc_pred = svc_grid.predict(features_test)
end_predicting = time()
print("Predicting time : {}".format(end_predicting - start_predicting))

svc_accuracy = accuracy_score(svc_pred, labels_test)
print('SVC accuracy score : {}'.format(svc_accuracy))
print "f1 score :", f1_score(svc_pred, labels_test)
print "precision score :", precision_score(svc_pred, labels_test)
print "recall score :", recall_score(svc_pred, labels_test)
svc_best_estimator = svc_grid.best_estimator_
print(svc_best_estimator)

test_classifier(nb_grid.best_estimator_, my_dataset, features_list)

# Verificando o resultado do novo feature no classificador final
test_features_list = ['poi',
                      'total_stock_value',
                      'exercised_stock_options',
                      'bonus',
                      'deferred_income',
                      'long_term_incentive',
                      'restricted_stock',
                      'salary',
                      'total_payments',
                      'other',
                      'shared_receipt_with_poi',
                      'fraction_from_this_person_to_poi']

print "\n=================================================================================="
print '''\nExaminado agora com as seguintes features:
* poi
* total_stock_value
* exercised_stock_options
* bonus
* deferred_income
* long_term_incentive
* restricted_stock
* salary
* total_payments
* other
* shared_receipt_with_poi 
* fraction_from_this_person_to_poi
'''
print "\n==============Resultado com features listado acima no classificador==============="

test_classifier(nb_grid.best_estimator_, my_dataset, test_features_list)

###Task 6: Dump your classifier, dataset, and features_list so anyone can
print "\n=============================dump_classifier_and_data============================="
dump_classifier_and_data(nb_grid.best_estimator_, my_dataset, features_list)
print "dump_classifier_and_data(nb_grid.best_estimator_, my_dataset, features_list)"
print "\nnb_grid.best_estimator_ = ", nb_grid.best_estimator_
print "\nfeatures_list = ", pprint.pprint(features_list)
