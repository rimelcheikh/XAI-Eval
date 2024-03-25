import tensorflow as tf
import numpy as np
import pickle
import os

from tensorflow.keras import layers, models
from keras.utils import plot_model
from keras.models import Model, load_model
#from keras.utils.generic_utils import get_custom_objects
from keras.layers import GlobalAveragePooling2D, Dense


from keras.applications.inception_v3 import InceptionV3 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
#from keras.optimizers import SGD, Adam
from keras.optimizers.legacy import SGD
from keras.utils import get_custom_objects

from keras.layers import Layer, Activation
from keras import backend as K
from keras import Sequential

from sklearn import svm
from joblib import dump, load




def DAP_model(targets, concepts, model_name, weights_matrix, mat_pd, X_train, y_train, y_train_0, X_valid, y_valid, idx_to_label_cifar, pretrained):
    
    
    
    
    #Training the Neural Architecture + Training SVR
    if pretrained: 
        model = load_model('./DAP/'+model_name)
        #model = load_model('./DAP/model')
        with open('./DAP/'+model_name+'/objects/extracted_features_train.pkl','rb') as f:
            extracted_features_train = pickle.load(f) 
        
        SVR = load('./SV/svr_linear_all.joblib')
        
            
            
        
    else:
        if model_name == 'vgg_16':
            base_model = VGG16(include_top=False)#, input_shape=(32, 32, 3))
        elif model_name == 'vgg_19':
            base_model = VGG19(include_top=False)#, input_shape=(32, 32, 3))
        
        # make all layers trainable
        for layer in base_model.layers:
            layer.trainable = True
            
        
        get_custom_objects().update({'custom_activation': Activation(attributes_layer_activation)})

        
        # add your head on top
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        """x = layers.Dense(len(concepts), activation='relu',trainable=False,use_bias=False)(x)
        predictions = layers.Dense(len(targets),trainable=False,use_bias=False,activation='softmax')(x)
        model = Model(base_model.input, predictions)"""
        
        x = CustomSVRActivationLayer((None,len(concepts)))(x)
       
        #x = layers.Dense(len(concepts), activation=Activation(attributes_layer_activation, name='SpecialActivation'),trainable=False,use_bias=False)(x)
        predictions = layers.Dense(len(targets),trainable=False,use_bias=False,activation='softmax')(x)
        model = Model(base_model.input, predictions)
        
        
        #model.layers[-1].set_weights([weights_matrix])
        
        #plot_model(model, to_file='./DAP_model.png', show_shapes=True)
        
        
        model.compile(optimizer=SGD(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, 
                            validation_data=(X_valid, y_valid))
        
        model.save('./DAP/'+model_name)
        
        
        #Prep features for SVR training
        inp = model.layers[0].input                                           # input placeholder
        outputs = model.layers[0].output          # all layer outputs
        functors = K.function([inp], [outputs])    # evaluation functions

        extracted_features_train = []
        for img in X_train:
            train = img.reshape((1,img.shape[0],img.shape[1],img.shape[2],))
            bn_outs = np.array(functors([train])).reshape(512)
            
            extracted_features_train.append(bn_outs)
            
            
        if not os.path.exists('./DAP/'+model_name+'/objects/'):
            os.makedirs('./DAP/'+model_name+'/objects/')
        
        with open('./DAP/'+model_name+'/objects/'+'/extracted_features_train.pkl','wb') as f:
            pickle.dump(extracted_features_train,f) 
         
        ##Prep labels (=annotations) for all instances to train SVR per attribute(neuron)
        svr_labels_train = {}
        for k in mat_pd.index:
            svr_labels_train[k] = []
            for ann in y_train_0:
                svr_labels_train[k].append(mat_pd[idx_to_label_cifar[ann[0]]].loc[k])
        
                
        
            
        with open('./DAP/'+model_name+'/objects/svr_labels_train.pkl','wb') as f:
            pickle.dump(svr_labels_train,f)
        
        SVR = train_SVR(extracted_features_train, svr_labels_train, model_name)
  
        
    return model, SVR






def run_testing_DAP(model, model_name, SVR, mat_pd, idx_to_label_cifar, X_test, y_test, y_test_0, pretrained):
    inp = model.layers[0].input                                           # input placeholder
    outputs = model.layers[0].output          # all layer outputs
    functors = K.function([inp], [outputs])  
    
    # 1. Extracting features with trained model
    if pretrained:
        with open('./DAP/'+model_name+'/objects/extracted_features_test.pkl','rb') as f:
            extracted_features_test = pickle.load(f) 
    
    else:
        extracted_features_test = []
        for img in X_test:
            test = img.reshape((1,img.shape[0],img.shape[1],img.shape[2],))
            bn_outs = np.array(functors([test])).reshape(512)
            
            extracted_features_test.append(bn_outs)
            
            
        with open('./DAP/'+model_name+'/objects/extracted_features_test.pkl','wb') as f:
            pickle.dump(extracted_features_test,f)
    
    
            
        
    # 2. Get SVR predictions

    ##Prep labels (=annotations) for all instances to get predictions with SVR p(a|x)
    svm_labels_test = {}
    for k in mat_pd.index:
        svm_labels_test[k] = []
        for ann in y_test_0:
            svm_labels_test[k].append(mat_pd[idx_to_label_cifar[ann[0]]].loc[k])


    with open('./DAP/'+model_name+'/objects/svm_labels_test.pkl','wb') as f:
        pickle.dump(svm_labels_test,f)
     
    ## {'concept_am':[score_img1,...,score_imgx]}
    X_svr_test = np.array(extracted_features_test)
    pred_per_concept = {}
    for c in SVR:
       predictor = SVR[c]
       pred_per_concept[c] = predictor.predict(X_svr_test)
       
    ## have preds as {'instance_x':[score_a1,...,score_aM]}
    pred_per_class_per_x = {}
    for k in np.unique(y_test_0):
        pred_per_class_per_x[idx_to_label_cifar[k]] = []
        for i in range(len(X_test)):
            tmp = []
            if y_test_0[i][0] == k:
                for c in pred_per_concept:
                    tmp.append(pred_per_concept[c][i])
                pred_per_class_per_x[idx_to_label_cifar[k]].append(tmp)
   
    if not os.path.exists('./SV/'+model_name+'/predictions/'):
        os.makedirs('./SV/'+model_name+'/predictions/')        
   
    
    with open('./SV/'+model_name+'/predictions/pred_per_class_per_x.pkl', 'wb') as f:
        pickle.dump(pred_per_class_per_x,f)
    with open('./SV/'+model_name+'/predictions/pred_per_concept.pkl', 'wb') as f:
        pickle.dump(pred_per_concept,f)




    return pred_per_class_per_x




def train_SVR(extracted_features_train, svr_labels_train, model_name):
    #SVR for concepts
    X_svr_train = np.array(extracted_features_train)
    
    reg_by_concept = {}

    for concept in svr_labels_train.keys():
        print('Training SVR for concept : ' + concept)
        y_svr_train = np.array(svr_labels_train[concept])
        #y_svr_test = np.array(svr_labels_train[concept])
        
        reg = svm.SVR(kernel='linear') 
        reg.fit(X_svr_train, y_svr_train)


        """parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[1,3,6,8,10]}
        reg = GridSearchCV(svm.SVR(), parameters)
        #GridSearchCV(estimator=svm.SVR(), param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
        reg.fit(X_svr_train, y_svr_train)"""
        
        reg_by_concept[concept] = reg
        
        
        dump(reg, './SV/'+model_name+'/svr_linear_'+concept+'.joblib')
    
    dump(reg_by_concept, './SV/'+model_name+'/svr_linear_all.joblib')
    





def attributes_layer_activation(x):
    
    SVR = load('./SV/svr_linear_all.joblib')
    res = []
    for c in SVR:
        res.append(SVR[c].predict(x))
    
    t = tf.constant(res, tf.float32)    
    
    return t
   
    with open('./pred_per_class_per_x.pkl','rb') as f:
        pred = pickle.load(f)
        
    
    
    #b = np.linalg.pinv(m.layers[-1].get_weights()).transpose().reshape((12,17))
    #a = np.array(m.layers[-2].get_weights()).reshape((512,12))
    #res = np.matmul(a,b)   

    return res    
    #return 
    



class CustomSVRActivationLayer(Layer):
    def __init__(self, o_s,**kwargs):
        super(CustomSVRActivationLayer, self).__init__(**kwargs)
        self.svr = None  # SVR model will be initialized in the build method
        self.o_s = o_s

    def build(self, input_shape):
        print('BUILD!!!!!!!!!!',input_shape)
        super(CustomSVRActivationLayer, self).build(input_shape)
        self.svr = load('./SV/svr_linear_all.joblib')

    def svr_function(self, inputs):
        # Convert SymbolicTensor to NumPy array
        inputs_np = inputs.numpy()

        # Assuming inputs_np is a NumPy array, adjust as needed
        flat_inputs = inputs_np#tf.keras.backend.batch_flatten(inputs_np)

        """predictions = []
        # Make predictions using the trained SVR model
        for i in range(flat_inputs.shape[0]):
            preds = []
            for c in self.svr:
                preds.append(self.svr[c].predict(flat_inputs[i].reshape(1, -1))[0])
                print('???',np.shape(preds),np.shape([preds]*512))
            predictions.append(preds)#([preds]*512)
            print('?????????????????????',np.shape(predictions))"""
            
        predictions = []
        # Make predictions using the trained SVR model
        for c in self.svr:
            preds = []
            for i in range(flat_inputs.shape[0]):
                preds.append(self.svr[c].predict(flat_inputs[i].reshape(1, -1))[0])
                #print('???',np.shape(preds),np.shape(preds*512))
            predictions.append(preds)#*512)#([preds]*512)
        predictions = np.array(predictions).transpose().tolist()
        #print(p)
        print(np.shape(predictions), flat_inputs.shape)
        print(tf.shape(tf.constant(predictions,tf.float32)),'____',tf.shape(tf.constant([predictions],tf.float32)))
        # Assuming you want to return a NumPy array, adjust as needed
        return tf.constant(predictions,tf.float32)#np.reshape(predictions, (flat_inputs.shape[0], -1))
    
    def call(self, inputs):
        # Assuming inputs is a 2D tensor, adjust as needed
        # running eager execution in Graph
        print('CALL!!!',inputs.shape)
        print('!!!!!!!!!', tf.py_function(self.svr_function, [inputs], tf.float32).shape)
        res = tf.py_function(self.svr_function, [inputs], tf.float32)
        res.set_shape(self.o_s)
        print('2 !!!!!!!!!', res.shape)
        return res


    def compute_output_shape(self, input_shape):
        return input_shape





    

class CustomSVRLayer(Layer):
  """``CustomLayer``."""
  def __init__(self, units, concepts, name="SVR_layer"):
    super().__init__(name=name)
    concepts = [c.split('-')[0] for c in concepts]
    self.concepts = concepts
    self.units = units
    self.SVR = load('./SV/svr_linear_all.joblib')
    
    
  def build(self, input_shape):
    input_dim = input_shape[-1]
    self.kernel = self.add_weight("kernel", shape=(input_dim, self.units), trainable=True)

    """w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.units),
                             dtype='float32'), trainable=True)"""
    
  def call(self, x):    
    preds_to_return = []
    for c in self.concepts:
        preds_to_return.append(self.SVR[c].predict(x))
    return tf.constant(preds_to_return, tf.float32)
      



    
class CustomPredLayer(Layer):
  """``CustomLayer``."""
  def __init__(self, name="pred_layer"):
    super().__init__(name=name)
    
  def build(self, input_shape):
    self.w = self.add_weight(
        shape=(1, 3),
        initializer="random_normal",
        trainable=True)
    
  def call(self, x):
   
        
    frames = tf.signal.frame(x, 3, 1)
    return tf.math.reduce_sum(tf.math.abs(frames - self.w), axis=-1)

