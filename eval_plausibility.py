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
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import r2_score


xai = 'tcav'
dataset = 'imagenet'  
model_name = "resnet_101"
concepts_dataset = 'broden'
rationales_dataset = 'awa'

num_random_exp = 10

alphas_cav = [0.01]
model_cav = 'logistic'

test_dir = model_name+'_'+dataset+'_'+concepts_dataset+'_'+rationales_dataset+'_'+model_cav+'_'+str(alphas_cav[0])
res_dir = './results/'+test_dir
data_dir = './data/'








if rationales_dataset == "apy":
    rationales_mat = apy_rationales()
    get_asso_strength = get_apy_asso_strength
    
    targets = [ 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'sheep', 'train', 'monkey', 'wolf', 'zebra', 
               'mug', 'bag','bicycle', 'bird',]
    concepts = ['tail', 'ear', 'nose', 'mouth', 'hair', 'face', 'eye', 'torso', 'hand', 'arm', 'leg', 'wing', 'wheel', 'door', 
                'headlight', 'taillight', 'engine', 'text', 'horn', 'saddle', 'leaf', 'flower', 'pot', 'screen', 'skin', 'metal', 'plastic', 'wood', 
                'glass', 'leather']#'beak', 'head'
    concepts = ['tail', 'mouth', 'hair', 'face', 'hand', 'arm', 'wing', 'wheel', 'door', 'headlight', 'taillight', 'engine', 'horn', 'saddle', 'flower', 'pot', 'wood', 'glass']
    targets = ['indigo bird','snowbird','speedboat','lifeboat','fireboat','soda bottle','beer bottle','wine bottle','water bottle','minibus','trolleybus',
            'school bus']
    targets = ['fireboat','hummingbird']
    targets = []
    for d in os.listdir(data_dir+'/imgs/apy_targets/targets'):
        if not len(os.listdir(data_dir+'/imgs/apy_targets/targets/'+d)) == 0:
            targets.append(d) 
    targets = ['dog','horse','train','tvmonitor','aeroplane','bicycle','bird', 'boat', 'bottle', 'bus', 'car', 
               'cat', 'chair']
    
    
if rationales_dataset == "awa":
    get_asso_strength = get_awa_asso_strength
    targets = ['elephant', 'squirrel', 'rabbit', 'wolf', 'buffalo', 'fox', 'leopard','gorilla','ox','chimpanzee','hamster','weasel','lion','tiger','hippopotamus','dalmatian','zebra',
               'otter','mouse','collie','beaver','skunk'] 

    targets = os.listdir(data_dir+'/imgs/awa_targets')

    concepts = ['ocean-s', 'desert-s', 'forest-s', 'water-s', 'cave-s', 'black-c', 'brown-c', 'white-c', 'blue-c', 'orange-c', 'red-c', 'yellow-c']
    rationales_mat = awa_rationales(concepts)

topk = 6 


if model_name == 'inceptionv3':
    bottleneck = ['mixed10']
elif model_name == 'resnet_101':
    bottleneck = ['conv5_block3_add']
elif model_name == 'vgg_16':
    bottleneck = ['block5_pool']


if not os.path.exists(res_dir):
    os.makedirs(res_dir)
    
with open(res_dir+'/'+rationales_dataset+'_params.txt', 'w') as f:
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
def spearmans_rank(exp,rationale):
    R_exp = rankdata(exp)
    R_rationale = rankdata(rationale)
    
    return np.corrcoef(R_exp, R_rationale)



def prep_tcav_rationales_vectors(topk_scores, method, rationales_dataset):
    sp_coeff_concepts = {}
    
    if rationales_dataset == "awa":
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
            
            
    
    
    if rationales_dataset == "apy":
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
            
            rationale_mean, tcav_mean = [], []
            for i in range(len(rationale)):
                rationale_mean.append(np.mean(rationale[i]))
                tcav_mean.append(np.mean(tcav[i]))
            
            sp_coeff_concepts[target] = spearmans_rank(tcav_mean, rationale_mean)[0][1]
        
        
        if not os.path.exists(res_dir+'/plausibility/'+rationales_dataset+'/'):
            os.makedirs(res_dir+'/plausibility/'+rationales_dataset+'/')
        with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_spearman_fixed_target.pkl', 'wb') as fp:
            pickle.dump(sp_coeff_concepts, fp)
            print('Spearman coeff on fixed targets saved successfully to file') 
        
        
        with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_spearman_fixed_target.pkl', 'rb') as fp:
            print('Results:', pickle.load(fp))
    
        return sp_coeff_concepts





def r2_sim(topk_scores, method, rationales_dataset):
    r2_sim = {}
    
    if rationales_dataset == "awa":
        for target in targets:
            rationale_dict = {}
            rationale = []
            topk_concepts = list(topk_scores[target].keys()) 
            print(target, topk_concepts)
            for c in topk_concepts:
                print(c)
                rationale_dict[c] = get_asso_strength(target,c, rationales_mat)/100
                rationale.append(get_asso_strength(target,c, rationales_mat)/100)
            
            tcav_dict = topk_scores[target] 
            tcav = []
            for v in topk_scores[target].keys():
                tcav.append(topk_scores[target][v])
            
            r2_sim[target] = r2_score(tcav, rationale)
        
    
        with open(res_dir+'/plausibility/'+method+'_'+str(topk)+'_R2_fixed_target.pkl', 'wb') as fp:
            pickle.dump(r2_sim, fp)
            print('Spearman coeff on fixed targets saved successfully to file') 
        
        
        with open(res_dir+'/plausibility/'+method+'_'+str(topk)+'_R2_fixed_target.pkl', 'rb') as fp:
            print('Results:', pickle.load(fp))
    
        return r2_sim
    
    
    if rationales_dataset == "apy":
        for target in targets:
            rationale_dict = {}
            rationale = []
            topk_concepts = list(topk_scores[target].keys()) 
            print(target, topk_concepts)
            for c in topk_concepts:
                rationale_dict[c] = get_asso_strength(target,c, rationales_mat)/100
                rationale.append(get_asso_strength(target,c, rationales_mat)/100)
            
            tcav_dict = topk_scores[target] 
            tcav = []
            for v in topk_scores[target].keys():
                tcav.append(topk_scores[target][v])
            
            rationale_mean, tcav_mean = [], []
            for i in range(len(rationale)):
                rationale_mean.append(np.mean(rationale[i]))
                tcav_mean.append(np.mean(tcav[i]))
            
            r2_sim[target] = r2_score(tcav_mean, rationale_mean)
        
        
        if not os.path.exists(res_dir+'/plausibility/'+rationales_dataset+'/'):
            os.makedirs(res_dir+'/plausibility/'+rationales_dataset+'/')
        with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_'+str(topk)+'_R2_fixed_target.pkl', 'wb') as fp:
            pickle.dump(r2_sim, fp)
            print('Cosine similarities on fixed targets saved successfully to file') 
        
        
        with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_'+str(topk)+'_R2_fixed_target.pkl', 'rb') as fp:
            print('Results:', pickle.load(fp))
    
        return r2_sim




def cosine_sim(topk_scores, method, rationales_dataset):
    cos_sim = {}
    
    if rationales_dataset == "awa":
        for target in targets:
            rationale_dict = {}
            rationale = []
            topk_concepts = list(topk_scores[target].keys()) 
            print(target, topk_concepts)
            for c in topk_concepts:
                print(c)
                rationale_dict[c] = get_asso_strength(target,c, rationales_mat)/100
                rationale.append(get_asso_strength(target,c, rationales_mat)/100)
            
            tcav_dict = topk_scores[target] 
            tcav = []
            for v in topk_scores[target].keys():
                tcav.append(topk_scores[target][v])
            
            cos_sim[target] = dot(tcav, rationale)/(norm(tcav)*norm(rationale))
        
    
        with open(res_dir+'/plausibility/'+method+'_'+str(topk)+'_cosine_fixed_target.pkl', 'wb') as fp:
            pickle.dump(cos_sim, fp)
            print('Spearman coeff on fixed targets saved successfully to file') 
        
        
        with open(res_dir+'/plausibility/'+method+'_'+str(topk)+'_cosine_fixed_target.pkl', 'rb') as fp:
            print('Results:', pickle.load(fp))
    
        return cos_sim
    
    
    if rationales_dataset == "apy":
        for target in targets:
            rationale_dict = {}
            rationale = []
            topk_concepts = list(topk_scores[target].keys()) 
            print(target, topk_concepts)
            for c in topk_concepts:
                rationale_dict[c] = get_asso_strength(target,c, rationales_mat)/100
                rationale.append(get_asso_strength(target,c, rationales_mat)/100)
            
            tcav_dict = topk_scores[target] 
            tcav = []
            for v in topk_scores[target].keys():
                tcav.append(topk_scores[target][v])
            
            rationale_mean, tcav_mean = [], []
            for i in range(len(rationale)):
                rationale_mean.append(np.mean(rationale[i]))
                tcav_mean.append(np.mean(tcav[i]))
            
            cos_sim[target] = dot(tcav_mean, rationale_mean)/(norm(tcav_mean)*norm(rationale_mean))
        
        
        if not os.path.exists(res_dir+'/plausibility/'+rationales_dataset+'/'):
            os.makedirs(res_dir+'/plausibility/'+rationales_dataset+'/')
        with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_'+str(topk)+'_cosine_fixed_target.pkl', 'wb') as fp:
            pickle.dump(cos_sim, fp)
            print('Cosine similarities on fixed targets saved successfully to file') 
        
        
        with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_'+str(topk)+'_cosine_fixed_target.pkl', 'rb') as fp:
            print('Results:', pickle.load(fp))
    
        return cos_sim



def euclidean_sim(topk_scores, method, rationales_dataset):
    euc_sim = {}
    
    if rationales_dataset == "awa":
        for target in targets:
            rationale_dict = {}
            rationale = []
            topk_concepts = list(topk_scores[target].keys()) 
            print(target, topk_concepts)
            for c in topk_concepts:
                print(c)
                rationale_dict[c] = get_asso_strength(target,c, rationales_mat)/100
                rationale.append(get_asso_strength(target,c, rationales_mat)/100)
            
            tcav_dict = topk_scores[target] 
            tcav = []
            for v in topk_scores[target].keys():
                tcav.append(topk_scores[target][v])
            
            euc_sim[target] = 1./(np.finfo(float).eps+np.sqrt(sum(pow(a-b,2) for a, b in zip(tcav, rationale))))
            euc_sim[target] = 1./(1+np.sqrt(sum(pow(a-b,2) for a, b in zip(tcav, rationale))))
        
    
        with open(res_dir+'/plausibility/'+method+'_'+str(topk)+'_euclidean_fixed_target.pkl', 'wb') as fp:
            pickle.dump(euc_sim, fp)
            print('Euclidean similarities on fixed targets saved successfully to file') 
        
        
        with open(res_dir+'/plausibility/'+method+'_'+str(topk)+'_euclidean_fixed_target.pkl', 'rb') as fp:
            print('Results:', pickle.load(fp))
    
        return euc_sim

    if rationales_dataset == "apy":
        for target in targets:
            rationale_dict = {}
            rationale = []
            topk_concepts = list(topk_scores[target].keys()) 
            print(target, topk_concepts)
            for c in topk_concepts:
                rationale_dict[c] = get_asso_strength(target,c, rationales_mat)
                rationale.append(get_asso_strength(target,c, rationales_mat))
            
            tcav_dict = topk_scores[target] 
            tcav = []
            for v in topk_scores[target].keys():
                tcav.append(topk_scores[target][v])
            
            rationale_mean, tcav_mean = [], []
            for i in range(len(rationale)):
                rationale_mean.append(np.mean(rationale[i]))
                tcav_mean.append(np.mean(tcav[i]))
            
            euc_sim[target] = 1./(np.finfo(float).eps+np.sqrt(sum(pow(a-b,2) for a, b in zip(tcav_mean, rationale_mean))))
            euc_sim[target] = 1./(1+np.sqrt(sum(pow(a-b,2) for a, b in zip(tcav_mean, rationale_mean))))
        
        
        if not os.path.exists(res_dir+'/plausibility/'+rationales_dataset+'/'):
            os.makedirs(res_dir+'/plausibility/'+rationales_dataset+'/')
        with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_'+str(topk)+'_euclidean_fixed_target.pkl', 'wb') as fp:
            pickle.dump(euc_sim, fp)
            print('Euclidean similarities on fixed targets saved successfully to file') 
        
        
        with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_'+str(topk)+'_euclidean_fixed_target.pkl', 'rb') as fp:
            print('Results:', pickle.load(fp))
    
        return euc_sim



#computing spearman when target is fixed 
def spearman_target_fixed(topk_scores, method, rationales_dataset):
    sp_coeff_concepts = {}
    
    if rationales_dataset == "awa":
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
        
    
        with open(res_dir+'/plausibility/'+method+'_'+str(topk)+'_spearman_fixed_target.pkl', 'wb') as fp:
            pickle.dump(sp_coeff_concepts, fp)
            print('Spearman coeff on fixed targets saved successfully to file') 
        
        
        with open(res_dir+'/plausibility/'+method+'_'+str(topk)+'_spearman_fixed_target.pkl', 'rb') as fp:
            print('Results:', pickle.load(fp))
    
        return sp_coeff_concepts
    
    
    if rationales_dataset == "apy":
        for target in targets:
            rationale_dict = {}
            rationale = []
            topk_concepts = list(topk_scores[target].keys()) 
            print(target, topk_concepts)
            for c in topk_concepts:
                rationale_dict[c] = get_asso_strength(target,c, rationales_mat)
                rationale.append(get_asso_strength(target,c, rationales_mat))
            
            tcav_dict = topk_scores[target] 
            tcav = []
            for v in topk_scores[target].keys():
                tcav.append(topk_scores[target][v])
            
            rationale_mean, tcav_mean = [], []
            for i in range(len(rationale)):
                rationale_mean.append(np.mean(rationale[i]))
                tcav_mean.append(np.mean(tcav[i]))
            
            sp_coeff_concepts[target] = spearmans_rank(tcav_mean, rationale_mean)[0][1]
        
        
        if not os.path.exists(res_dir+'/plausibility/'+rationales_dataset+'/'):
            os.makedirs(res_dir+'/plausibility/'+rationales_dataset+'/')
        with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_'+str(topk)+'_spearman_fixed_target.pkl', 'wb') as fp:
            pickle.dump(sp_coeff_concepts, fp)
            print('Spearman coeff on fixed targets saved successfully to file') 
        
        
        with open(res_dir+'/plausibility/'+rationales_dataset+'/'+method+'_'+str(topk)+'_spearman_fixed_target.pkl', 'rb') as fp:
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


#for topk in range(1,len(concepts)):
if xai == 'cce':
    run_eval_cce(targets, concepts, dataset, rationales_dataset, model_name, bottleneck, num_random_exp, alphas_cav, model_cav, res_dir+'/cce/', data_dir)
    #cce_apy_concepts = os.listdir(data_dir+'/broden/apy')
    topk_cce_scores, cce_scores = prep_cce_res(res_dir+'/cce/', targets, concepts, topk)
    
    euclidean_sim(topk_cce_scores, 'cce', rationales_dataset)
    cosine_sim(topk_cce_scores, 'cce', rationales_dataset)
    spearman_target_fixed(topk_cce_scores,'cce',rationales_dataset)
    r2_sim(topk_cce_scores,'cce',rationales_dataset)
    #spearman_target_concept_fixed(topk_cce_scores,'cce',rationales_dataset)

if xai == 'tcav':
    run_eval_tcav(targets, concepts, dataset, model_name, bottleneck, num_random_exp, alphas_cav, model_cav, res_dir + '/tcav/', data_dir)
    topk_tcav_scores, tcav_scores = prep_tcav_res(res_dir+'/tcav/', targets, concepts, topk,rationales_dataset)
    
    euclidean_sim(topk_tcav_scores, 'tcav', rationales_dataset)
    cosine_sim(topk_tcav_scores, 'tcav', rationales_dataset)
    spearman_target_fixed(topk_tcav_scores,'tcav',rationales_dataset)
    r2_sim(topk_tcav_scores,'tcav',rationales_dataset)
    #spearman_target_concept_fixed(topk_tcav_scores,'tcav',rationales_dataset)
    




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
