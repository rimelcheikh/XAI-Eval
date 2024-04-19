import numpy as np
import tensorflow as tf
import keras.backend as K

from tensorflow.keras import layers, models
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.applications.vgg19 import VGG19
from keras.layers import GlobalAveragePooling2D, Dense, Layer
from keras.optimizers.legacy import SGD

from joblib import load


class DAPModel(Model):
    def __init__(self,model,weights_matrix, mat_pd, targets_idx):
        super(DAPModel, self).__init__()
        #self.model = model     
        self.backbone_layer_0 = model.layers[0]
        self.backbone_layer_1 = model.layers[1]
        self.backbone_layer_2 = model.layers[2]
        self.backbone_layer_3 = model.layers[3]
        self.backbone_layer_4 = model.layers[4]
        self.backbone_layer_5 = model.layers[5]
        self.backbone_layer_6 = model.layers[6]
        self.backbone_layer_7 = model.layers[7]
        self.backbone_layer_8 = model.layers[8]
        self.backbone_layer_9 = model.layers[9]
        self.backbone_layer_10 = model.layers[10]
        self.backbone_layer_11 = model.layers[11]
        self.backbone_layer_12 = model.layers[12]
        self.backbone_layer_13 = model.layers[13]
        self.backbone_layer_14 = model.layers[14]
        self.backbone_layer_15 = model.layers[15]
        self.backbone_layer_16 = model.layers[16]
        self.backbone_layer_17 = model.layers[17]
        self.backbone_layer_18 = model.layers[18]
        self.backbone_layer_19 = model.layers[19]
        self.backbone_layer_20 = model.layers[20]
        self.backbone_layer_21 = model.layers[21]
        #self.dense_layer = Dense(512)
        self.GlobalAveragePooling2D =  GlobalAveragePooling2D()
        self.attributes_layer = CustomSVRActivationLayer(mat_pd,targets_idx, tf.convert_to_tensor(weights_matrix,dtype=tf.float32)) #TODO : define output dim       
        self.class_layer = CustomLabelsLayer(tf.convert_to_tensor(weights_matrix,dtype=tf.float32))
        self.weights_matrix = tf.convert_to_tensor(weights_matrix,dtype=tf.float32)
        self.mat_pd = mat_pd
        self.targets_idx = targets_idx
        self.true_label = None
        self.training = None
        #self.out = self.call(self.input_layer)
    

    def get_config(self):
        config = super(DAPModel, self).get_config()
        # Add custom configuration parameters here if needed
        return config

    
    def cal(self,inputs):
        
        #return inputs#self.model(inputs)
        #x = self.model(inputs)
        x = self.backbone_layer_0(inputs)
        x = self.backbone_layer_1(x)
        x = self.backbone_layer_2(x)
        x = self.backbone_layer_3(x)
        x = self.backbone_layer_4(x)
        x = self.backbone_layer_5(x)
        x = self.backbone_layer_6(x)
        x = self.backbone_layer_7(x)
        x = self.backbone_layer_8(x)
        x = self.backbone_layer_9(x)
        x = self.backbone_layer_10(x)
        x = self.backbone_layer_11(x)
        x = self.backbone_layer_12(x)
        x = self.backbone_layer_13(x)
        x = self.backbone_layer_14(x)
        x = self.backbone_layer_15(x)
        x = self.backbone_layer_16(x)
        x = self.backbone_layer_17(x)
        x = self.backbone_layer_18(x)
        x = self.backbone_layer_19(x)
        x = self.backbone_layer_20(x)
        x = self.backbone_layer_21(x)
        #x = self.dense_layer(x)
        x = self.GlobalAveragePooling2D(x)  #shape=(64,512)
        x = self.attributes_layer(x,self.true_label,self.training)   #[64, 12]
        return self.class_layer(x)   #[64, 17]
    
    def call(self,inputs,training=None,true_label=None):
       #return inputs#self.model(inputs)
       #x = self.model(inputs
       """x = self.backbone_layer_0(inputs)
       x = self.backbone_layer_1(x)
       x = self.backbone_layer_2(x)
       x = self.backbone_layer_3(x)
       x = self.backbone_layer_4(x)
       x = self.backbone_layer_5(x)
       x = self.backbone_layer_6(x)
       x = self.backbone_layer_7(x)
       x = self.backbone_layer_8(x)
       x = self.backbone_layer_9(x)
       x = self.backbone_layer_10(x)
       x = self.backbone_layer_11(x)
       x = self.backbone_layer_12(x)
       x = self.backbone_layer_13(x)
       x = self.backbone_layer_14(x)
       x = self.backbone_layer_15(x)
       x = self.backbone_layer_16(x)
       x = self.backbone_layer_17(x)
       x = self.backbone_layer_18(x)
       x = self.backbone_layer_19(x)
       x = self.backbone_layer_20(x)
       x = self.backbone_layer_21(x)
       x = self.GlobalAveragePooling2D(x)  #shape=(64,512)
       return x"""
       """x = self.attributes_layer(x,self.true_label,self.training)   #[64, 12]
       print('okk')
       return self.class_layer(x)   #[64, 17]"""
       return inputs
    

    
    
    def convert_to_ragged(self,instances):
        if type(instances) is tf.RaggedTensor:
              out = instances
        elif type(instances) is tf.Tensor:
              out = tf.RaggedTensor.from_tensor(instances)
        else:
              out = tf.ragged.constant(np.asarray(instances))
        return out

      
  
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
    

        with tf.GradientTape() as tape:
            self.true_label = y
            self.training = True

            self(x)
            y_pred = self.cal(x)#,True, y)  # Forward pass, contains return of call function
            
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(y, y_pred, sample_weight=None, regularization_losses=self.losses)

        #np.shape(self.layers[0].get_weights()[1])
        #self.layers[1].get_weights()[0].shape
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=None)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}






    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        
        self.training = False
        # Compute predictions
        self(x)
        y_pred = self.cal(x)#, None, training=False)
        
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    
    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        
        """self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.out
        )"""

        super(DAPModel, self).build(input_shape)
        

class CustomLabelsLayer(Layer):
    def __init__(self,weights_matrix,**kwargs):
        super(CustomLabelsLayer, self).__init__(**kwargs)
        self.weights_matrix = weights_matrix
         #self.output_shape = output_shape #à définir


    def call(self, inputs):
        return Dense(17)(inputs)
        
        w = self.weights_matrix#tf.convert_to_tensor(self.weights_matrix,dtype=tf.float32)
        w_i = tf.linalg.pinv(w).transpose()
        res = tf.linalg.matmul(inputs,w_i)  
        return res
    
        
        
        """svr_preds = inputs.numpy()
        w = np.linalg.pinv(self.weights_matrix).transpose()
        res = np.matmul(svr_preds,w)            
        outputs = res
        return tf.constant(outputs,tf.float32)"""

    def compute_output_shape(self, input_shape):
        return input_shape       
    
    def build(self, input_shape):
        super(CustomLabelsLayer, self).build(input_shape)
        self.w = self.add_weight(name='kernel',
                                      shape=(int(input_shape[1]),17, int(input_shape[0])),
                                      trainable=False)
        self.b = self.add_weight(name='bias',shape=(int(input_shape[0]),), initializer="zeros", trainable=False)

        
        #super().build(input_shape)
        
        


class CustomSVRActivationLayer(Layer):
    def __init__(self,mat_pd,targets_idx,weights_matrix,**kwargs):
        super(CustomSVRActivationLayer, self).__init__(**kwargs)
        self.svr = load('./SV/svr_linear_all.joblib')
        self.alpha = None
        self.reg = None
        self.x_avg = {}
        self.y_avg = {}
        self.Sxy = {}
        self.Sx = {} 
        self.n = {}
        self.mat_pd = mat_pd
        self.targets_idx = targets_idx
        self.weights_matrix = weights_matrix
        self.reg_model = {}


    def reg_function(self, inputs, true_label, training):
        # Define the shape of the 2D tensor
        rows = self.weights_matrix.shape[0] 
        cols = len(true_label)
        
        # Initialize an empty tensor with zeros
        reg_labels_train = tf.zeros((rows, cols))

        for j in range(rows):
            for i in range(cols):            
                # Example: Assigning a value based on row and column indices
                reg_labels_train = tf.tensor_scatter_nd_update(
                    reg_labels_train,
                    indices=[[j, i]],  # Update element at (i, j)
                    updates=[self.weights_matrix[j][true_label[i]][0]]  # Example update value
                )
            
    
        res = tf.zeros(((len(reg_labels_train)),len(true_label)))

        
        if training:            
            for i in range(len(reg_labels_train)):
                if i not in self.reg_model.keys():
                    self.reg_model[i] = OnlineLinearRegressor(inputs.shape[0])
                
                reg_model = self.reg_model[i]
                self.new_x_avg[i], self.new_y_avg[i], self.new_Sxy[i], self.new_Sx[i], self.new_n[i] = self.reg_model[i].fit(self.x_avg[i],
                                      self.y_avg[i],
                                      self.Sxy[i],
                                      self.Sx[i],
                                      self.n[i],
                                      inputs, 
                                      reg_labels_train[i])
                #self.reg_model[i] = self.reg
                
                upd = self.reg_model[i].predict(inputs)
                
                res = tf.tensor_scatter_nd_update(
                    res,
                    indices=[[i]],  # Update element at (i, j)
                    updates=[upd]  # Example update value
                )
        else:           
            for i in range(len(reg_labels_train)): 
                upd = self.reg_model[i].predict(inputs)
                res = tf.tensor_scatter_nd_update(
                    res,
                    indices=[[i]],  # Update element at (i, j)
                    updates=[upd]  # Example update value
                )       
                
            
        return res.transpose()

    
    def call(self, inputs, true_label, training):       
        return Dense(12)(inputs)
        print('okkk')
        res = self.reg_function(inputs, true_label, training)
        print('----',res)
        return res


    def compute_output_shape(self, input_shape):
        return input_shape
    
    
    def build(self, input_shape):
        #à vérifier la dimension des poids et de input_shape
        
        self.w = self.add_weight(name='kernel',
                                      shape=(int(input_shape[1]),12, int(input_shape[0])),
                                      trainable=False)
        self.b = self.add_weight(name='bias',shape=(int(input_shape[0]),), initializer="zeros", trainable=False)

        super(CustomSVRActivationLayer, self).build(input_shape)
        #super().build(input_shape)
        print('ok')




class OnlineLinearRegressor:
    def __init__(self, n_samples,learning_rate=0.01):
        self.learning_rate = learning_rate
        self.beta = None
        self.alpha = None
        self.x_avg = tf.zeros((n_samples,))
        self.y_avg = tf.zeros((n_samples,))
        self.Sxy = tf.zeros((n_samples,))
        self.Sx = tf.zeros((n_samples,)) 
        self.n = 0.0
        

    def fit(self, x_avg,y_avg,Sxy,Sx,n,new_x,new_y):
        
        
        new_n = n + len(new_x)
        new_x_avg = (x_avg*n + tf.math.reduce_sum(new_x,1))/new_n
        new_y_avg = (y_avg*n + tf.math.reduce_sum(new_y))/new_n
        
        if n > 0:
            x_star = (x_avg*tf.math.sqrt(n) + new_x_avg*tf.math.sqrt(new_n))/(tf.math.sqrt(n)+tf.math.sqrt(new_n))
            y_star = (y_avg*tf.math.sqrt(n) + new_y_avg*tf.math.sqrt(new_n))/(tf.math.sqrt(n)+tf.math.sqrt(new_n))
        elif n == 0:
            x_star = new_x_avg
            y_star = new_y_avg
        else:
            raise ValueError
        
        x_star_expanded = tf.expand_dims(x_star, axis=1)
        y_star_expanded = tf.expand_dims(y_star, axis=0)
        new_Sx =  Sx + tf.math.reduce_sum((new_x-x_star_expanded)**2,1)
        new_Sxy = Sxy + tf.math.reduce_sum(((new_x-x_star_expanded)*(tf.expand_dims(new_y, axis=1)-y_star_expanded)),1) #TODO : check correctness

        beta = new_Sxy/new_Sx   
        alpha = new_y_avg - beta * new_x_avg

        # Update parameters
        self.beta = beta
        self.alpha = alpha
        
        self.x_avg = new_x_avg
        self.y_avg = new_y_avg
        self.Sxy = new_Sxy
        self.Sx = new_Sx 
        self.n = new_n
        
        return new_x_avg, new_y_avg, new_Sxy, new_Sx, new_n 

        
    def predict(self, X):
        res = tf.zeros((X.shape[0]))
        for i in range(X.shape[0]):
            upd = 0
            for j in range(X.shape[1]):
                upd += X[i][j]*self.beta[i] 
            res = tf.tensor_scatter_nd_update(
                res,
                indices=[[i]],  # Update element at (i, j)
                updates=[upd]  # Example update value
            )
            
        return res + self.alpha






def get_DAP_model(targets, concepts, model_name, weights_matrix, mat_pd, X_train, y_train, y_train_0, X_valid, y_valid, idx_to_label_cifar, targets_idx, pretrained):
    #tf.compat.v1.enable_eager_execution()
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    
    base_model = VGG19(include_top=False)
     
    # make all layers trainable
    for layer in base_model.layers:
        layer.trainable = True
    
    # add your head on top
    x = base_model.output
    model = Model(base_model.input)
    
    
    model = DAPModel(base_model, weights_matrix, mat_pd, idx_to_label_cifar)
    
    #model.build(input_shape=(X_train.shape[1:]))

    
    #model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer=SGD(), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'],run_eagerly=True)
    
    from datetime import datetime
    from packaging import version
    from tensorflow import keras
    logdir="./DAP/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    """graph = tf.compat.v1.get_default_graph()
    with tf.compat.v1.Graph().as_default():
        writer = tf.compat.v1.summary.FileWriter(logdir, graph)
        writer.close()"""
        
    """with tf.summary.create_file_writer(logdir).as_default():
        tf.summary.trace_on(graph=True, profiler=True)
        dummy_input = tf.zeros((1, *X_train.shape[1:]))  # Create dummy input
        tf.summary.trace_export(name="my_trace", step=0, profiler_outdir=logdir)"""
    
    # Start tracing the graph
    """with tf.summary.create_file_writer(logdir).as_default():
        tf.summary.trace_on(graph=True, profiler=True)

        # Execute a forward pass to generate the graph
        dummy_input = tf.zeros((1, *X_train.shape[1:]))  # Create dummy input
        _ = model(dummy_input)

        # Export the graph
        tf.summary.trace_export(name="my_trace", step=0, profiler_outdir=logdir)"""
    
    history = model.fit(X_train[:32], y_train[:32], epochs=2, batch_size=16,
                        validation_data=(X_valid[:8], y_valid[:8]), 
                        )
    
    
    model.save('./DAP/'+model_name+'_2')
    
    
    return model 