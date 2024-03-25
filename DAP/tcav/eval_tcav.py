# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:57:14 2023

@author: rielcheikh
"""

import sys, os
import tensorflow as tf
import numpy as np

from tcav.run_tcav import run_tcav
from tcav.tcav.model_custom import run_tcav_custom 
import tcav.tcav.utils as utils
import tcav.tcav.utils_plot as utils_plot
import pickle
from os.path import exists

from keras.models import load_model
from keras.optimizers import SGD, Adam


#import tcav.repro_check as repro_check

import pdb
#pdb.set_trace()


#eval_save_dir = "./tmp/" + user + '/' + project_name

result = {}

tf.compat.v1.enable_eager_execution() 




def run_eval_tcav(model, targets, concepts, dataset, model_name, weights_matrix, bottleneck, num_random_exp, alphas, model_cav, res_dir,data_dir, pred_per_class_per_x=None):
    
    """adjusted_model= tf.keras.models.clone_model(model)
    #adjusted_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    adjusted_model.set_weights(model.get_weights()) """
    
    for target in targets:
        """svr_preds_target = pred_per_class_per_x[target]
        svr_preds_target_avg = np.mean(svr_preds_target,axis=0)
        svr_preds_target_avg = [svr_preds_target_avg]*512
        
        
        adjusted_model.layers[-2].set_weights([np.array(svr_preds_target_avg)/100])
        adjusted_model.save('./DAP/adjusted_'+model_name+'_'+target)"""
        
        
        
        #try:
        project_name = 'tcav_test_'+str(target)
        working_dir = res_dir + project_name
    
        if not exists(working_dir):
            os.makedirs(working_dir)
        
        if not exists(working_dir+'/tcav_res_'+target+'.pkl'):
            run_tcav_custom(model, target, concepts, dataset, bottleneck, model_name, working_dir, data_dir, num_random_exp, alphas, model_cav)

        else:
            print(working_dir+'/tcav_res_'+target+'.pkl'+'  already computed')

                        

    
    
    
    
    

    
    
   






