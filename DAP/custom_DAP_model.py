import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.applications.vgg19 import VGG19
from keras.layers import GlobalAveragePooling2D, Dense, Layer
from keras.optimizers.legacy import SGD

from joblib import load


class DAPModel(Model):
    def __init__(self,model,weights_matrix):
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
        self.GlobalAveragePooling2D =  GlobalAveragePooling2D()
        self.attributes_layer = CustomSVRActivationLayer() #TODO : define output dim       
        self.class_layer = CustomLabelsLayer(weights_matrix)
        self.weights_matrix = weights_matrix

    """def build(self, input_shape):
        super(DAPModel, self).build(input_shape)"""

    

    
    """def __call__(self,inputs):
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
        x = self.GlobalAveragePooling2D(x)  #shape=(64,512)
        x = self.attributes_layer(x)   #[64, 12]
        return self.class_layer(x)   #[64, 17]"""
    
    def call(self,inputs):
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
        x = self.GlobalAveragePooling2D(x)  #shape=(64,512)
        x = self.attributes_layer(x)   #[64, 12]
        return self.class_layer(x)   #[64, 17]
    
    """def call(self,inputs):
        return self(inputs)"""
    
    
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
        
        """base_model = VGG19(include_top=False)
        for layer in base_model.layers:
            layer.trainable = True
        #x = self.convert_to_ragged(x)
        x = base_model(x)"""
        
        """x = self.model(x)
        
        #x = GlobalAveragePooling2D()(x)
        x = self.attributes_layer(x)
        x = self.class_layer(x)"""
        #x = Dense(17,trainable=False,use_bias=False,activation='softmax')(x)
        
        #trainable_vars = self.trainable_variables

        with tf.GradientTape() as tape:
            #tape.watch(self.trainable_variables)
            y_pred = self(x)#, training=True)  # Forward pass, contains return of call function
            #y_pred = tf.ragged.boolean_mask(y_pred,masks) # truncate according to masks
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(y, y_pred, sample_weight=None, regularization_losses=self.losses)

        """self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables[1], tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight=None)"""
        
        #np.shape(self.layers[0].get_weights()[1])
        #self.layers[1].get_weights()[0].shape
        gradients = tape.gradient(loss, self.trainable_variables[:-2])

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
        
        #x = self.convert_to_ragged(x)
        #x = VGG19(include_top=False)(x)
        """x = self.model(x)
        #x = GlobalAveragePooling2D()(x)
        x = self.attributes_layer(x)
        x = self.class_layer(x)"""
        #x = Dense(17,trainable=False,use_bias=False,activation='softmax')(x)
        


        # Compute predictions
        y_pred = self(x)#, training=False)
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
         #self.output_shape = output_shape #à définir


    def call(self, inputs):
        """res = []
        for i in range(inputs.shape[0]):"""
        svr_preds = inputs.numpy()
        w = np.linalg.pinv(self.weights_matrix).transpose()
        res = np.matmul(svr_preds,w)            
        outputs = res
        return tf.constant(outputs,tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape       
    
    def build(self, input_shape):
        
        self.w = self.add_weight(name='kernel',
                                      shape=(int(input_shape[1]),17, int(input_shape[0])),
                                      trainable=True)
        self.b = self.add_weight(shape=(int(input_shape[0]),), initializer="zeros", trainable=False)

        #super(CustomLabelsLayer, self).build(input_shape)
        super().build(input_shape)
        
        


class CustomSVRActivationLayer(Layer):
    def __init__(self,**kwargs):
        super(CustomSVRActivationLayer, self).__init__(**kwargs)
        self.svr = load('./SV/svr_linear_all.joblib')
        

    """def build(self, input_shape):
        print('BUILD!!!!!!!!!!',input_shape)
        super(CustomSVRActivationLayer, self).build(input_shape)
        self.svr = load('./DAP/SV/svr_linear_all.joblib')"""
        
    """def build(self, input_shape):
        super(CustomSVRActivationLayer, self).build(input_shape)"""

    def svr_function(self, inputs):
        # Convert Tensor to NumPy array __call__
        inputs_np = inputs.numpy()

        # Assuming inputs_np is a NumPy array, adjust as needed
        flat_inputs = inputs_np#tf.keras.backend.batch_flatten(inputs_np)

            
        predictions = []
        # Make predictions using the trained SVR model
        for c in self.svr:
            preds = []
            for i in range(flat_inputs.shape[0]):
                preds.append(self.svr[c].predict(flat_inputs[i].reshape(1, -1))[0]/100)
            predictions.append(preds)
        predictions = np.array(predictions).transpose().tolist()
        #TODO : negative predictions ?
        return tf.constant(predictions,tf.float32)#np.reshape(predictions, (flat_inputs.shape[0], -1))
    
    def call(self, inputs):
        print("BEFORE",inputs)
        inputs = tf.stop_gradient(inputs)
        print("AFTER",inputs)
        res = self.svr_function(inputs)
        
        return res


    def compute_output_shape(self, input_shape):
        return input_shape
    
    
    def build(self, input_shape):
        #à vérifier la dimension des poids et de input_shape
        self.w = self.add_weight(name='kernel',
                                      shape=(int(input_shape[1]),12, int(input_shape[0])),
                                      trainable=True)
        self.b = self.add_weight(shape=(int(input_shape[0]),), initializer="zeros", trainable=False)

        #super(CustomSVRActivationLayer, self).build(input_shape)
        super().build(input_shape)



def get_DAP_model(targets, concepts, model_name, weights_matrix, mat_pd, X_train, y_train, y_train_0, X_valid, y_valid, idx_to_label_cifar, pretrained):
    

    base_model = VGG19(include_top=False)
     
    # make all layers trainable
    for layer in base_model.layers:
        layer.trainable = False
    
    # add your head on top
    x = base_model.output
    model = Model(base_model.input)
    
    
    model = DAPModel(base_model, weights_matrix)
    
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
    
    history = model.fit(X_train[:64], y_train[:64], epochs=1, batch_size=64,
                        validation_data=(X_valid[:32], y_valid[:32]), 
    callbacks=[tensorboard_callback])
    
    
    model.save('./DAP/'+model_name)
    
    
    return model 