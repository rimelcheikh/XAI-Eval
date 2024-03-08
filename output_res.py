import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import os


xai = 'tcav'
model = 'inceptionv3_'
dataset = 'imagenet_'
concept_dataset = 'broden_'
rationales_dataset = 'awa_'
cav_model = 'logistic_'
cav_alpha = '0.01'
topk = 18

sim_measure = 'spearman'

list_sim = {}

for sim_measure in ['spearman','cosine', 'euclidean','R2']:
    list_sim[sim_measure] = []
    for topk in range(1,12):
        res_dir = './results/'+model+dataset+concept_dataset+rationales_dataset+cav_model+cav_alpha+'/plausibility/'
        
        
        with open(res_dir+xai+'_'+str(topk)+'_'+sim_measure+'_fixed_target.pkl', 'rb') as f:
            res = pickle.load(f)
        
        res_avg = []
        for k in list(res.keys()):
            if not math.isnan(res[k]):
                res_avg.append(np.abs(res[k]))
            
        list_sim[sim_measure].append(np.mean(res_avg))

if not os.path.exists('./results/figures/'):
    os.makedirs('./results/figures/')       

for i in list_sim.keys():
    plt.plot(list_sim[i], label=i)
    plt.legend()
    plt.title(xai+"_"+model+rationales_dataset)
    plt.xlabel('Top k')
    plt.savefig('./results/figures/'+xai+"_"+model+rationales_dataset+'.png')