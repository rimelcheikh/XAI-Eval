from tcav.eval_tcav import run_eval_tcav
from cce.eval_cce import run_eval_cce
from data.AwA2base.awa_rationales import get_awa_asso_strength, awa_rationales
from data.aPY.aPY_rationales import get_apy_asso_strength, apy_rationales

from tcav.prep_tcav_res import prep_tcav_res
from cce.prep_cce_res import prep_cce_res

from scipy.stats import rankdata
import numpy as np
import pickle
import os


xai = 'tcav'
dataset = 'imagenet'  
model_name = "inceptionv3"
concepts_dataset = 'broden'
rationales_dataset = 'apy'

num_random_exp = 10

alphas_cav = [0.01]
model_cav = 'logistic'

test_dir = model_name+'_'+dataset+'_'+concepts_dataset+'_'+model_cav+'_'+str(alphas_cav[0])
res_dir = './results/'+test_dir
data_dir = './data/'




topk = 5


#awa
"""targets = ['elephant', 'squirrel', 'rabbit', 'wolf', 'buffalo', 'fox', 'leopard','gorilla','ox','chimpanzee','hamster','weasel','lion','tiger','hippopotamus','dalmatian','zebra',
           'otter','mouse','collie','beaver','skunk'] 

targets = os.listdir(data_dir+'/imgs/awa_targets')

concepts = ['ocean-s', 'desert-s', 'forest-s', 'water-s', 'cave-s', 'black-c', 'brown-c', 'white-c', 'blue-c', 'orange-c', 'red-c', 'yellow-c']"""


#apy
targets = [ 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'sheep', 'train', 'monkey', 'wolf', 'zebra', 
           'mug', 'bag','bicycle', 'bird',]
concepts = ['tail', 'head', 'ear', 'nose', 'mouth', 'hair', 'face', 'eye', 'torso', 'hand', 'arm', 'leg', 'wing', 'wheel', 'door', 
            'headlight', 'taillight', 'engine', 'text', 'horn', 'saddle', 'leaf', 'flower', 'pot', 'screen', 'skin', 'metal', 'plastic', 'wood', 
            'glass', 'leather']#'beak', 
targets = ['indigo bird','snowbird','speedboat','lifeboat','fireboat','soda bottle','beer bottle','wine bottle','water bottle','minibus','trolleybus',
        'school bus']
targets = ['fireboat','hummingbird']



if rationales_dataset == "apy":
    rationales_mat = apy_rationales()
    get_asso_strength = get_apy_asso_strength
if rationales_dataset == "awa":
    rationales_mat = awa_rationales(concepts)
    get_asso_strength = get_awa_asso_strength



if model_name == 'inceptionv3':
    bottleneck = ['mixed10']
elif model_name == 'resnet_101':
    bottleneck = ['conv5_block3_add']
elif model_name == 'vgg_16':
    bottleneck = ['block5_pool']


if not os.path.exists(res_dir):
    os.makedirs(res_dir)
    
with open(res_dir+'/'+model_name+'_params.txt', 'w') as f:
    f.write('dataset : '+dataset+'\n')
    f.write('black box : '+model_name+'\n')
    f.write('bottleneck : '+bottleneck[0]+'\n')
    f.write('concepts dataset : ' + concepts_dataset+'\n')
    f.write('rationales dataset : ' + rationales_dataset+'\n')
    f.write('number of random examples : ' + str(num_random_exp)+'\n')
    f.write('classifier for CAVs and alpha : ' + model_cav + ' (' + str(alphas_cav[0])+')\n')
    f.write('K for top-K : '+str(topk)+'\n')
    f.write('targets : ')
    for t in targets:
        f.write(t+' ')
    f.write('\n')
    f.write('concepts : ')
    for c in concepts:
        f.write(c+' ')
    
  








# Computing Spearman's rank correlation coefficient between the sensitivity and the predictio scores
def spearmans_rank(exp, pred):
    R_exp = rankdata(exp)
    R_pred = rankdata(pred)
    
    return np.corrcoef(R_exp, R_pred)



#computing spearman when target is fixed 
def spearman_target_fixed(topk_scores, method, rationales_dataset):
    sp_coeff_concepts = {}
    for target in targets:
        rationale_dict = {}
        rationale = []
        topk_concepts = list(topk_scores[target].keys()) 
        print(target, topk_concepts)
        for c in topk_concepts:
            print(c)
            rationale_dict[c] = get_asso_strength(target,c, rationales_mat)
            rationale.append(get_asso_strength(target,c, rationales_mat))
        
        tcav_dict = topk_scores[target] 
        tcav = []
        for v in topk_scores[target].keys():
            tcav.append(topk_scores[target][v])
        
        sp_coeff_concepts[target] = spearmans_rank(rationale, tcav)[0][1]
    

    with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_spearman_fixed_target.pkl', 'wb') as fp:
        pickle.dump(sp_coeff_concepts, fp)
        print('Spearman coeff on fixed targets saved successfully to file') 
    
    
    with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_spearman_fixed_target.pkl', 'rb') as fp:
        print('Results:', pickle.load(fp))

    return sp_coeff_concepts



#computing spearman when (target,concept) are fixed
def spearman_target_concept_fixed(topk_scores, method, rationales_dataset):
    sp_coeff = {}
    tcav_dict = {}
    rationale_dict = {}
    
    for target in targets:
        rationale = {}
        topk_concepts = list(topk_scores[target].keys()) 
        for c in topk_concepts:
            #rationale_dict[c] = get_asso_strength(target,c.split('-')[0])
            rationale[c] = (get_asso_strength(target,c,rationales_mat))
        
        tcav_dict[target] = topk_scores[target]
        rationale_dict[target] = rationale
        
    r, s = {}, {}
    r_vect, s_vect = [], []
    for t in targets:
        topk_concepts = list(topk_scores[t].keys()) 
        for c in topk_concepts:
            r[t,c] = rationale_dict[t][c]
            s[t,c] = tcav_dict[t][c]
            r_vect.append(rationale_dict[t][c])
            s_vect.append(tcav_dict[t][c])
    sp_coeff = spearmans_rank(r_vect, s_vect)[0][1]
    
    with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_spearman_fixed_target_concept.pkl', 'wb') as fp:
        pickle.dump(sp_coeff, fp)
        print('Spearman coeff on fixed targets saved successfully to file')
    
    
    with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_spearman_fixed_target_concept.pkl', 'rb') as fp:
        print('Results:', pickle.load(fp))

    return sp_coeff


if not os.path.exists(res_dir+'/plausibility/'):
    os.makedirs(res_dir+'/plausibility/')
    
    
#TODO : rationales_dataset

if xai == 'cce':
    run_eval_cce(targets, concepts, dataset, rationales_dataset, model_name, bottleneck, num_random_exp, alphas_cav, model_cav, res_dir+'/cce/', data_dir)
    #cce_apy_concepts = os.listdir(data_dir+'/broden/apy')
    topk_cce_scores, cce_scores = prep_cce_res(res_dir+'/cce/', targets, concepts, topk)
    
    spearman_target_fixed(topk_cce_scores,'cce',rationales_dataset)
    spearman_target_concept_fixed(topk_cce_scores,'cce',rationales_dataset)

if xai == 'tcav':
    run_eval_tcav(targets, concepts, dataset, model_name, bottleneck, num_random_exp, alphas_cav, model_cav, res_dir + '/tcav/', data_dir)
    topk_tcav_scores, tcav_scores = prep_tcav_res(res_dir+'/tcav/', targets, concepts, topk)
    
    spearman_target_fixed(topk_tcav_scores,'tcav',rationales_dataset)
    spearman_target_concept_fixed(topk_tcav_scores,'tcav',rationales_dataset)





"""#computing spearman when concept is fixed
sp_coeff_targets = {}
for c in concepts:
    rationale_dict = {}
    rationale = []
    tcav = []
    for t in targets:
        #TODO
        rationale_dict[t] = get_asso_strength(t,c.split('-')[0])
        rationale.append(get_asso_strength(t,c.split('-')[0]))
    
        tcav.append(class_tcav_score[t][c])
        
    sp_coeff_targets[c] = spearmans_rank(rationale, tcav)[0][1]


with open("./results/tcav/plausibility/" + user +'/result_fixed_concept.pkl', 'wb') as fp:
    pickle.dump(sp_coeff_targets, fp)
    print('Spearman coeff on fixed concepts saved successfully to file')


with open("./results/tcav/plausibility/" + user +'/result_fixed_concept.pkl', 'rb') as fp:
    print('Results:', pickle.load(fp))"""
