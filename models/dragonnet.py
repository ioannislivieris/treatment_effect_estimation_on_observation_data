import os
from tabnanny import verbose
import time
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, Dense, Layer, Concatenate
from models.losses import (dragonnet_loss_binarycross,
                           regression_loss,
                           binary_classification_loss, treatment_accuracy, track_epsilon, make_tarreg_loss)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks  import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau

class EpsilonLayer(Layer):
    """
    Custom Keras layer that outputs a trainable epsilon term.
    This can be used to model uncertainty or residual adjustment in Dragonnet.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
      
    def build(self, input_shape):
        """
        Initialize a trainable weight called 'epsilon'.
        Args:
            input_shape (TensorShape): Shape of the input tensor
        """
        # Single trainable scalar weight
        self.epsilon = self.add_weight(
            name='epsilon',
            shape=(1, 1),
            initializer='RandomNormal',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Forward pass for the layer.
        Args:
            inputs (Tensor): Input tensor of shape (batch_size, features)
        Returns:
            Tensor: Epsilon tensor broadcasted to batch size
        """
        # Broadcast epsilon to match batch size
        batch_size = tf.shape(inputs)[0]
        return tf.ones((batch_size, 1), dtype=tf.float32) * self.epsilon


def make_dragonnet(input_dim: int, reg_l2: float) -> Model:
    """
    Creates a Dragonnet model with three outputs:
    - y0_predictions: potential outcome under control
    - y1_predictions: potential outcome under treatment
    - t_predictions: propensity score
    - epsilon: trainable residual term

    Args:
        input_dim (int): Number of features in the input
        reg_l2 (float): L2 regularization strength for hypothesis layers

    Returns:
        tf.keras.Model: Compiled Dragonnet model
    """
    # Reproducibility
    tf.random.set_seed(42)

    # Input layer
    inputs = Input(shape=(input_dim,), name='input_features')

    # Shared representation layers
    x = Dense(200, activation='elu', kernel_initializer='he_normal', name='rep_dense_1')(inputs)
    x = Dense(200, activation='elu', kernel_initializer='he_normal', name='rep_dense_2')(x)
    x = Dense(200, activation='elu', kernel_initializer='he_normal', name='rep_dense_3')(x)

    # Treatment prediction head
    t_predictions = Dense(1, activation='sigmoid', name='treatment_pred')(x)

    # Hypothesis heads for potential outcomes
    def hypothesis_head(name_prefix: str, x_input: tf.Tensor) -> tf.Tensor:
        """
        Creates a hypothesis head for potential outcomes
        """
        h = Dense(100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),
                  name=f'{name_prefix}_hidden_1')(x_input)
        h = Dense(100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),
                  name=f'{name_prefix}_hidden_2')(h)
        return Dense(1, activation=None, kernel_regularizer=regularizers.l2(reg_l2),
                     name=f'{name_prefix}_pred')(h)

    y0_predictions = hypothesis_head('y0', x)
    y1_predictions = hypothesis_head('y1', x)

    # Epsilon layer for residuals / uncertainty
    epsilon = EpsilonLayer(name='epsilon')(t_predictions)

    # Concatenate all outputs
    outputs = Concatenate(name='dragonnet_output')([y0_predictions, y1_predictions, t_predictions, epsilon])

    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name='Dragonnet')

    return model




class Dragonnet:
    """
    Wrapper class for the Dragonnet model.
    """

    def __init__(self):
        """
        Initializes the Dragonnet model.

        Args:
            input_dim (int): Number of features in the input
            reg_l2 (float): L2 regularization strength for hypothesis layers
        """
        self.model = None
        
    def fit(self, X:np.ndarray=None, T:np.ndarray=None, Y:np.ndarray=None, parameters = {}):
        reg_l2 = parameters.get('reg_l2', 1e-2)
        targeted_regularization = parameters.get('targeted_regularization', True)
        knob_loss = parameters.get('knob_loss', dragonnet_loss_binarycross)
        loss_ratio = parameters.get('loss_ratio', 1.)
        validation_split = parameters.get('validation_split', 0.1)
        batch_size = parameters.get('batch_size', 16)
        verbose = parameters.get('verbose', False)
        # Setup Dragonnet model
        self.model = make_dragonnet(X.shape[1], reg_l2)
        # Define loss and metrics
        metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]
        loss = make_tarreg_loss(ratio=loss_ratio, dragonnet_loss=knob_loss) if targeted_regularization else knob_loss    
        # Create outputs for DragonNet (concatenate Y & T)
        yt = np.concatenate([Y.reshape(-1,1), T.reshape(-1,1)], axis = 1)


        # *** Training - Phase I ***
        # Compile network
        self.model.compile(optimizer = Adam(learning_rate=1e-3), loss=loss, metrics=metrics)
        # Setup callbacks
        callbacks = [TerminateOnNaN(),
                        EarlyStopping(monitor   = 'val_loss', 
                                    patience  = 3, 
                                    min_delta = 1e-6),
                        ReduceLROnPlateau(monitor   = 'loss', 
                                        factor    = 0.5, 
                                        patience  = 5, 
                                        verbose   = verbose, 
                                        mode      = 'auto', 
                                        min_delta = 1e-8, 
                                        cooldown  = 0, 
                                        min_lr    = 0)]
        # Training
        start_time = time.time()
        self.model.fit(X, yt, 
                        callbacks        = callbacks,
                        validation_split = validation_split,
                        epochs           = 100,
                        batch_size       = batch_size, 
                        verbose          = verbose)
        print("[INFO] Training - Phase I - Time %.2f secs" % (time.time() - start_time) )


        # *** Training - Phase II ***
        # Compile network
        self.model.compile(optimizer = SGD(learning_rate=1e-5, momentum=0.9, nesterov=True, clipnorm=1.0), 
                            loss      = loss,
                            metrics   = metrics)
        # Setup callbacks
        callbacks = [TerminateOnNaN(),
                        EarlyStopping(monitor   = 'val_loss', 
                                    patience  = 40, 
                                    min_delta = 1e-6),
                        ReduceLROnPlateau(monitor   = 'loss', 
                                        factor    = 0.5, 
                                        patience  = 5, 
                                        verbose   = verbose, 
                                        mode      = 'auto',
                                        min_delta = 0., 
                                        cooldown  = 0, 
                                        min_lr    = 0)
        ]
        # Training
        start_time = time.time()
        self.model.fit(X, yt, 
                        callbacks        = callbacks,
                        validation_split = validation_split,
                        epochs           = 300,
                        batch_size       = batch_size, 
                        verbose          = verbose)
        print("[INFO] Training - Phase II - Time %.2f secs" % (time.time() - start_time) )


    def predict(self, X:np.ndarray=None):
        """
        Predict potential outcomes and propensity scores.

        Parameters
        ----------
        X : array-like
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        dict
            Dictionary containing:
            - 'y_0_hat': Predicted outcome under control.
            - 'y_1_hat': Predicted outcome under treatment.
            - 'propensity_score': Estimated probability of treatment assignment.
        """
        # Get model predictions
        preds = self.model.predict(X)
        
        # Return dictionary of predictions
        return {
            'y_0_hat':  preds[:, 0],
            'y_1_hat': preds[:, 1],
            'propensity_score': preds[:, 2],
        }