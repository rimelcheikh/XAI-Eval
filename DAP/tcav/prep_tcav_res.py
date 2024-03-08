import pickle


def prep_tcav_res(res_dir, targets, concepts, topk, rationales_dataset):

    if rationales_dataset == "awa":
        class_tcav_score = {}
        topk_class_tcav_score = {}
        for target in targets:
           
            with open(res_dir+'/tcav_test_'+target+'/tcav_res_'+target+'.pkl', 'rb') as f:
                tcav_results = pickle.load(f)
                
                class_tcav_score[target] = {}
                topk_class_tcav_score[target] = {}
                for concept in concepts:
                    #search for random batch that gives max TCAV score (i_up) for concept
                    for i in range(len(tcav_results)):
                        if (tcav_results[i]['cav_concept'] == concept):# and (tcav_results[i]['cav_accuracies'][concept] == 1):
                            res = tcav_results[i]                    
                            try : 
                                if(class_tcav_score[target][concept] < res['i_up']):
                                    class_tcav_score[target][concept] = res['i_up']
                            except:
                                class_tcav_score[target][concept] = res['i_up']
                
                topk_concepts = sorted(class_tcav_score[target],key=class_tcav_score[target].__getitem__, reverse=True)[:topk]
                for c in topk_concepts:
                    topk_class_tcav_score[target][c] = class_tcav_score[target][c]
                            
        
        
        
        
        return topk_class_tcav_score, class_tcav_score
    
    
    
        
    if rationales_dataset == "apy":


        class_tcav_score = {}
        imgs_scores = {}
        topk_class_imgs_score = {}
        for target in targets:
           
            with open(res_dir+'/tcav_test_'+target+'/tcav_res_'+target+'.pkl', 'rb') as f:
                tcav_results = pickle.load(f)
                
                class_tcav_score[target] = {}
                imgs_scores[target] = {}
                topk_class_imgs_score[target] = {}
                for concept in concepts:
                    #search for random batch that gives max TCAV score (i_up) for concept
                    for i in range(len(tcav_results)):
                        if (tcav_results[i]['cav_concept'] == concept):# and (tcav_results[i]['cav_accuracies'][concept] == 1):
                            res = tcav_results[i]                    
                            try : 
                                if(class_tcav_score[target][concept] < res['i_up']):
                                    class_tcav_score[target][concept] = res['i_up']
                                    imgs_scores[target][concept] = res['per_img_tcav']
                            except:
                                class_tcav_score[target][concept] = res['i_up']
                                imgs_scores[target][concept] = res['per_img_tcav']
                
                topk_concepts = sorted(class_tcav_score[target],key=class_tcav_score[target].__getitem__, reverse=True)[:topk]
                
                for c in topk_concepts:
                    topk_class_imgs_score[target][c] = imgs_scores[target][c]
                    
                for k in topk_class_imgs_score.keys():
                    for kk in topk_class_imgs_score[k].keys():
                        for i in range(len(topk_class_imgs_score[k][kk])):
                            try:
                                topk_class_imgs_score[k][kk][i] = not topk_class_imgs_score[k][kk][i][0]
                            except: 
                                topk_class_imgs_score[k][kk][i] = not topk_class_imgs_score[k][kk][i]
        
        
        return topk_class_imgs_score, imgs_scores




"""for i in range(len(data)):
    if 'random' not in data[i]['cav_concept']:
        print(data[i]['cav_concept'], data[i]['cav_accuracies'][data[i]['cav_concept']], data[i]['i_up'])"""
        
"""for k in list(topk_class_tcav_score.keys()):
    print(k, topk_class_tcav_score[k])
    print('_______________')"""