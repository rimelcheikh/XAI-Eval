import os
import tensorflow as tf

import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model as model
import tcav.tcav as tcav
import tcav.utils as utils
import tcav.utils_plot as utils_plot


# This is the name of your model wrapper (InceptionV3 and GoogleNet are provided in model.py)
model_to_run = 'GoogleNet'  
user = 'rielcheikh'

# the name of the parent directory that results are stored (only if you want to cache)
project_name = 'tcav_test'
working_dir = "./tmp/" + user + '/' + project_name
source_dir = "./tcav/tcav_examples/image_models/imagenet/downloaded_data/"

# where activations are stored (only if your act_gen_wrapper does so)
activation_dir =  working_dir + '/activations/'

# where CAVs are stored. 
# You can say None if you don't wish to store any.
cav_dir = working_dir + '/cavs/'

bottlenecks = [ 'mixed4c']  # @param 
      
utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(cav_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs. 
alphas = [0.1]   

target = 'zebra'#'zebra'  
concepts = ["striped","dotted","zigzagged"] 

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
num_random_exp=3
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

utils_plot.plot_results(results,working_dir, num_random_exp=num_random_exp)


