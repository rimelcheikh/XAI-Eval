import tensorflow as tf
import numpy as np
import pickle
import os
import pandas as pd



from keras import backend as K
from tensorflow.keras import datasets

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from joblib import dump, load

from tcav.eval_tcav import run_eval_tcav
from DAP_model import DAP_model, run_testing_DAP

from custom_DAP_model_3 import get_DAP_model

pretrained = True


(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


with open('../data/CIFAR100/cifar100_fine_labels.txt') as f:
    targets_list = f.read().splitlines()


#targets_list = os.listdir('../data/imgs/awa_targets')
concepts = ['ocean-s', 'desert-s', 'forest-s', 'water-s', 'cave-s', 'black-c', 'brown-c', 'white-c', 'blue-c', 'orange-c', 'red-c', 'yellow-c']
      

weights_matrix = np.loadtxt('../data/AwA2base/Animals_with_Attributes2/predicate-matrix-continuous.txt').transpose()
classes = np.loadtxt('../data/AwA2base/Animals_with_Attributes2/classes.txt',dtype=str, usecols=1)
attributes = np.loadtxt('../data/AwA2base/Animals_with_Attributes2/predicates.txt',dtype=str, usecols=1)
mat_pd = pd.DataFrame(weights_matrix,index=attributes,columns=classes)


#get classes that are common to AwA and CIFAR AND and their CIFAR indexes
targets = []
ind = 0
ind_list = []
for t in targets_list:
    if t in classes:
        targets.append(t)
        ind_list.append(ind)
    ind +=1


#Making dict to map CIFAR class index to class name
idx_to_label_cifar = {}
label_to_idx_awa = {}
j = 0
for i in ind_list:
    idx_to_label_cifar[i] = targets[j]
    j+=1
    
    


#Get CIFAR index of images that are in desired classes
y_train_idx, y_test_idx = [], []   
for i in ind_list:
    y_train_idx.append(np.where(train_labels == i)[0])
    y_test_idx.append(np.where(test_labels == i)[0])

#Flattening the above result
i_train_idx_c, i_test_idx_c = [], []
for i in range(len(y_train_idx)):
    for j in range(np.shape(y_train_idx[i])[0]):
        i_train_idx_c.append(y_train_idx[i][j])
for i in range(len(y_test_idx)):
    for j in range(np.shape(y_test_idx[i])[0]):
        i_test_idx_c.append(y_test_idx[i][j])


#Keeping images and CIFAR labels of the desired classes
X_train, X_test = train_images[i_train_idx_c], test_images[i_test_idx_c]
y_train_0, y_test_0 = train_labels[i_train_idx_c], test_labels[i_test_idx_c]



        
for t in mat_pd.columns:
    if t.lower() not in targets:
        mat_pd = mat_pd.drop(t.lower(), axis=1)


#Get AwA indexes of classes
targets_idx = {}
for t in mat_pd.columns:
    targets_idx[t] = list(mat_pd.columns).index(t)


concepts_list = [c.split('-')[0] for c in concepts]
#Keeping lines that are Broden concepts
for c in mat_pd.index:
    if c.lower() not in concepts_list:
        mat_pd = mat_pd.drop(c.lower(), axis=0)

weights_matrix = np.array(mat_pd/100)


#Label images with AwA indexes
y_train, y_test = np.array([[0]]*len(y_train_0)), np.array([[0]]*len(y_test_0))
for i in range(len(y_train)):
    y_train[i][0] = targets_idx[idx_to_label_cifar[y_train_0[i][0]]]
for i in range(len(y_test)):
    y_test[i][0] = targets_idx[idx_to_label_cifar[y_test_0[i][0]]]





#Prep validation set by splitting X_train
X_train_tmp = X_train
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,random_state=10) 
X_train, X_valid, y_train_0, y_valid_0 = train_test_split(X_train_tmp, y_train_0, test_size=0.2,random_state=10) 





    

    




#Compute TCAV
dataset = 'CIFAR100' 
model_name = "vgg_19"
concepts_dataset = 'broden'
rationales_dataset = 'awa'

num_random_exp = 10

alphas_cav = [0.1]
model_cav = 'logistic'

test_dir = model_name+'_'+dataset+'_'+concepts_dataset+'_'+rationales_dataset+'_'+model_cav+'_'+str(alphas_cav[0])
res_dir = './results/'+test_dir
data_dir = '../data/'
bottleneck = ['block5_pool']




model = get_DAP_model(targets, concepts, model_name, weights_matrix, mat_pd, X_train, y_train, y_train_0, X_valid, y_valid, idx_to_label_cifar, targets_idx, pretrained)



#model, SVR = DAP_model(targets, concepts, model_name, weights_matrix, mat_pd, X_train, y_train, y_train_0, X_valid, y_valid, idx_to_label_cifar, targets_idx,pretrained)


#pred_per_x = run_testing_DAP(model, model_name, SVR, mat_pd, idx_to_label_cifar, X_test, y_test, y_test_0, pretrained)
    

"""for layer in model.layers:
    layer.trainable = True"""


run_eval_tcav(model, targets, concepts, dataset, model_name, weights_matrix, bottleneck, num_random_exp, alphas_cav, model_cav, res_dir + '/tcav/', data_dir)#, pred_per_x)
    

















