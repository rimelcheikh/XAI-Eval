import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.applications.vgg19 import VGG19
from keras.layers import GlobalAveragePooling2D, Dense, Layer
from keras.optimizers.legacy import SGD
from keras import backend as K


from joblib import load


class ModelPadder(Model):
    def __init__(self,model,weights_matrix):
        super(ModelPadder, self).__init__()
        self.model = model
        self.attributes_layer = CustomSVRActivationLayer()
        #TODO
        self.class_layer = CustomLabelsLayer(weights_matrix)
        self.weights_matrix = weights_matrix

    def build(self, input_shape):
        super(ModelPadder, self).build(input_shape)

    

    
    def call(self,inputs):
        x = self.model(inputs)
        x = self.attributes_layer(x)
        x = self.class_layer(x)
        return x
    

    
    
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
            #tape.watch(self.trainable_variables)
            y_pred = self(x, training=True)  # Forward pass, contains return of call function
            #y_pred = tf.ragged.boolean_mask(y_pred,masks) # truncate according to masks
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(y, y_pred, sample_weight=None, regularization_losses=self.losses)

        """self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables[1], tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight=None)"""
        
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=None)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


        """with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)"""



        """ # Compute gradients
        #TODO
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=None)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}"""




    def test_step(self, data):
        # Unpack the data
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        
    

        # Compute predictions
        y_pred = self(x, training=False)
        #y_pred = tf.ragged.boolean_mask(y_pred,masks)
        
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    
    def compute_output_shape(self, input_shape):
        return input_shape



class CustomLabelsLayer(Layer):
    def __init__(self,weights_matrix,**kwargs):
        super(CustomLabelsLayer, self).__init__(**kwargs)
        self.weights_matrix = weights_matrix


    def call(self, inputs):
        """res = []
        for i in range(inputs.shape[0]):"""
        svr_preds = inputs.numpy()
        w = np.linalg.pinv(self.weights_matrix).transpose()
        res = np.matmul(svr_preds,w)            
        #outputs = np.argmax(res,axis=1)
        #outputs = outputs.reshape((len(outputs),1))
        outputs = res
        return tf.constant(outputs,tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape        
        
        


class CustomSVRActivationLayer(Layer):
    def __init__(self,**kwargs):
        super(CustomSVRActivationLayer, self).__init__(**kwargs)
        self.svr = load('./DAP/SV/svr_linear_all.joblib')
        #self.o_s = o_s

    """def build(self, input_shape):
        print('BUILD!!!!!!!!!!',input_shape)
        super(CustomSVRActivationLayer, self).build(input_shape)
        self.svr = load('./SV/svr_linear_all.joblib')"""

    def svr_function(self, inputs):
        

            
        predictions = []
        # Make predictions using the trained SVR model
        for c in self.svr:
            preds = []
            for i in range(inputs.shape[3]):
                preds.append(inputs[i])
            predictions.append(preds)
            print(type(predictions))
        predictions = np.array(predictions).transpose().tolist()
        
        """inputs_np = inputs.numpy()

        # Assuming inputs_np is a NumPy array, adjust as needed
        flat_inputs = inputs_np

            
        predictions = []
        # Make predictions using the trained SVR model
        for c in self.svr:
            preds = []
            for i in range(flat_inputs.shape[0]):
                preds.append(self.svr[c].predict(flat_inputs[i].reshape(1, -1))[0]/100)
            predictions.append(preds)
        predictions = np.array(predictions).transpose().tolist()
        #TODO : negative predictions ?
        return tf.constant(predictions,tf.float32)"""
    
    def call(self, inputs):
        inputs = tf.stop_gradient(inputs)
        print(inputs, inputs.shape)
        res = self.svr_function(inputs)
        
        return res

    def compute_output_shape(self, input_shape):
        return input_shape



def get_DAP_model(targets, concepts, model_name, weights_matrix, mat_pd, X_train, y_train, y_train_0, X_valid, y_valid, idx_to_label_cifar, pretrained):
    
    base_model = VGG19(include_top=False)

    # Make all layers trainable
    for layer in base_model.layers:
        layer.trainable = True
        
    inputs = tf.keras.Input(shape=X_train.shape[1:])
    x = base_model(inputs)
    x = CustomSVRActivationLayer()(x)
    outputs = CustomLabelsLayer(weights_matrix)(x)
    model = tf.keras.Model(inputs, outputs)

     
    
    
    #model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer=SGD(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy (from_logits=False),
                  metrics=['accuracy'],  
                  run_eagerly=True)
    
    from datetime import datetime
    from packaging import version
    from tensorflow import keras
    logdir="./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    """graph = tf.compat.v1.get_default_graph()
    with tf.compat.v1.Graph().as_default():
        writer = tf.compat.v1.summary.FileWriter(logdir, graph)
        writer.close()"""
        
    with tf.summary.create_file_writer(logdir).as_default():
        tf.summary.trace_on(graph=True, profiler=True)
        dummy_input = tf.zeros((1, *X_train.shape[1:]))  # Create dummy input
        tf.summary.trace_export(name="my_trace", step=0, profiler_outdir=logdir)

    
    history = model.fit(X_train[:64], y_train[:64], epochs=2, batch_size=16,
                        validation_data=(X_valid[:32], y_valid[:32]), 
    callbacks=[tensorboard_callback])
    
    
    model.save('./DAP/'+model_name)
    
    
    return model 