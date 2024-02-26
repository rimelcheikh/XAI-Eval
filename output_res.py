import numpy as np
import pickle
import math


xai = 'tcav'
model = 'inceptionv3_'
dataset = 'imagenet_'
concept_dataset = 'broden_'
rationales_dataset = 'awa_'
cav_model = 'logistic_'
cav_alpha = '0.01'

sim_measure = 'cosine'

res_dir = './results/'+model+dataset+concept_dataset+rationales_dataset+cav_model+cav_alpha+'/plausibility/'


with open(res_dir+xai+'_'+sim_measure+'_fixed_target.pkl', 'rb') as f:
    res = pickle.load(f)

res_avg = []
for k in list(res.keys()):
    if not math.isnan(res[k]):
        res_avg.append(np.abs(res[k]))
    
np.mean(res_avg)