import os
import tensorflow as tf

import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model as model
import tcav.tcav as tcav
import tcav.utils as utils
import tcav.utils_plot as utils_plot
#import tcav.repro_check as repro_check

import pdb
#pdb.set_trace()

# This is the name of your model wrapper (InceptionV3 and GoogleNet are provided in model.py)
model_to_run = 'c'  
user = 'test4'

result = {}

tf.compat.v1.enable_eager_execution() 


# the name of the parent directory that results are stored (only if you want to cache)
for target in ['tiger']:#,'cheetah','zebra','lion','tiger']:#'leopard']:
    project_name = 'tcav_test_'+str(target)
    working_dir = "./tmp/" + user + '/' + project_name
    source_dir = "./tcav/tcav_examples/image_models/imagenet/data/"
    
    # where activations are stored (only if your act_gen_wrapper does so)
    activation_dir =  working_dir + '/activations/'
    
    # where CAVs are stored. 
    # You can say None if you don't wish to store any.
    cav_dir = working_dir + '/cavs/'
    
    bottlenecks = ['mixed3a']#,'mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']  # @param 
          
    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(working_dir)
    utils.make_dir_if_not_exists(cav_dir)
    
    # this is a regularizer penalty parameter for linear classifier to get CAVs. 
    alphas = [0.1]   
    
    #target = 'zebra'#'zebra'  
    concepts = ["striped"]#,"dotted"] 
    
    # Create TensorFlow session.
    sess = utils.create_session(interactive=True)
    
    # GRAPH_PATH is where the trained model is stored.
    GRAPH_PATH = source_dir + "/inception5h/tensorflow_inception_graph.pb"
    LABEL_PATH = source_dir + "/inception5h/imagenet_comp_graph_label_strings.txt"
    
    
    mymodel = model.GoogleNetWrapper_public(sess,
                                            GRAPH_PATH,
                                            LABEL_PATH)
    
    act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, 
                                                     max_examples=100)
    
    
    import absl
    absl.logging.set_verbosity(1)
    num_random_exp=2
    ## only running num_random_exp = 10 to save some time. The paper number are reported for 500 random runs. 
    mytcav = tcav.TCAV(sess,
                       target,
                       concepts,
                       bottlenecks,
                       act_generator,
                       alphas,
                       cav_dir=cav_dir,
                       num_random_exp=num_random_exp)
    print ('This may take a while... Go get coffee!')
    results = mytcav.run(run_parallel=False)
    print ('done!')
        
    #result.append(results)
    result[target] = results
    
    # Computing Spearman's rank correlation coefficient between the sensitivity and the predictio scores
    
    sp_coeff = {}
    for concept in concepts :
        sp_coeff[concept] = {}
        for bn in bottlenecks:
            exp_scores, pred_scores = utils.get_exp_and_pred_scores(results, target, concept, bn)
            sp_coeff[concept][bn] = utils.spearmans_rank(exp_scores, pred_scores)[0][1]
    
    utils_plot.plot_results(results,sp_coeff, working_dir, num_random_exp=num_random_exp)


#repro_check.repro_check(result)







