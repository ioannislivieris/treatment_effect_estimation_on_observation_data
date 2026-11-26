import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import binary_accuracy


def binary_classification_loss(concat_true, concat_pred):
    """
    Computes the binary cross-entropy loss for treatment prediction (propensity score).

    Args:
        concat_true (tf.Tensor): Tensor of shape (batch_size, 2+), [y_true, t_true, ...]
        concat_pred (tf.Tensor): Tensor of shape (batch_size, 3+), [y0_pred, y1_pred, t_pred, epsilon]

    Returns:
        tf.Tensor: Scalar loss
    """
    # Extract treatment labels and predictions
    t_true = tf.cast(
        concat_true[:, 1], tf.float32
    )  # cast to float32 to avoid dtype mismatch
    t_pred = tf.cast(concat_pred[:, 2], tf.float32)

    # Stabilize predictions to avoid log(0)
    t_pred = tf.clip_by_value(t_pred, 1e-3, 1.0 - 1e-3)

    # Compute binary crossentropy per sample
    return tf.reduce_sum(tf.keras.losses.binary_crossentropy(t_true, t_pred))


def regression_loss(concat_true, concat_pred):
    """
    Compute the Dragonnet regression loss for potential outcomes.

    Args:
        concat_true (tf.Tensor): Tensor of shape (batch_size, 2), [y_true, t_true]
        concat_pred (tf.Tensor): Tensor of shape (batch_size, 2), [y0_pred, y1_pred]

    Returns:
        tf.Tensor: Scalar loss
    """
    # Ensure everything is float32
    y_true = tf.cast(concat_true[:, 0], tf.float32)
    t_true = tf.cast(concat_true[:, 1], tf.float32)

    y0_pred = tf.cast(concat_pred[:, 0], tf.float32)
    y1_pred = tf.cast(concat_pred[:, 1], tf.float32)

    # Weighted squared error for control (t=0) and treatment (t=1)
    loss0 = tf.reduce_sum((1.0 - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1


def dragonnet_loss_binarycross(concat_true, concat_pred):
    """
    Computes the combined Dragonnet loss as the sum of regression loss
    (for potential outcomes) and binary classification loss (for treatment prediction).

    Args:
        concat_true (tf.Tensor): Tensor of shape (batch_size, 2) containing
                                 [y_true, t_true].
        concat_pred (tf.Tensor): Tensor of shape (batch_size, 4) containing
                                 [y0_pred, y1_pred, t_pred, epsilon].

    Returns:
        tf.Tensor: Scalar loss representing the combined regression + treatment loss.
    """
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(
        concat_true, concat_pred
    )

def treatment_accuracy(concat_true, concat_pred):
    """
    Computes the accuracy of treatment assignment predictions.

    Args:
        concat_true (tf.Tensor): Tensor of shape (batch_size, 2) containing
                                 [y_true, t_true].
        concat_pred (tf.Tensor): Tensor of shape (batch_size, 4) containing
                                 [y0_pred, y1_pred, t_pred, epsilon].

    Returns:
        tf.Tensor: Scalar representing the fraction of correct treatment predictions.
    """
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)


def track_epsilon(concat_true, concat_pred):
    """
    Computes the mean absolute value of the epsilon layer's outputs.
    Useful to monitor the magnitude of the targeted regularization term during training.

    Args:
        concat_true (tf.Tensor): Tensor of shape (batch_size, 2) containing
                                 [y_true, t_true].
        concat_pred (tf.Tensor): Tensor of shape (batch_size, 4) containing
                                 [y0_pred, y1_pred, t_pred, epsilon].

    Returns:
        tf.Tensor: Scalar representing the mean absolute epsilon.
    """
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


def make_tarreg_loss(ratio=1.0, dragonnet_loss=dragonnet_loss_binarycross):
    """
    Constructs a targeted regularization (TarReg) loss function for Dragonnet.
    This augments the base Dragonnet loss with a term that adjusts the outcomes
    using the epsilon layer to reduce bias in treatment effect estimation.

    Args:
        ratio (float, optional): Weight of the targeted regularization term. Defaults to 1.0.
        dragonnet_loss (callable, optional): Base Dragonnet loss function to augment.
                                             Defaults to dragonnet_loss_binarycross.

    Returns:
        callable: A loss function that can be used in `model.compile()`. The returned
                  function takes `(concat_true, concat_pred)` and returns a scalar loss.
    """
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        """
        Targeted regularization loss for Dragonnet (inner function).

        Args:
            concat_true (tf.Tensor): Tensor of shape (batch_size, 2) containing [y_true, t_true].
            concat_pred (tf.Tensor): Tensor of shape (batch_size, 4) containing
                                     [y0_pred, y1_pred, t_pred, epsilon].

        Returns:
            tf.Tensor: Scalar loss incorporating base Dragonnet loss + targeted regularization.
        """
        # Base Dragonnet loss
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        # Cast tensors to float32 for safe computation
        y_true = tf.cast(concat_true[:, 0], tf.float32)
        t_true = tf.cast(concat_true[:, 1], tf.float32)
        y0_pred = tf.cast(concat_pred[:, 0], tf.float32)
        y1_pred = tf.cast(concat_pred[:, 1], tf.float32)
        t_pred = tf.cast(concat_pred[:, 2], tf.float32)  # propensity head
        epsilons = concat_pred[:, 3]

        # Clip predicted treatment probabilities to avoid log(0) / division by zero
        t_pred = tf.clip_by_value(t_pred, 1e-3, 1-1e-3)

        # Predicted outcome under observed treatment
        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        # Clever covariate (IPW style)
        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        # Perturbed outcome
        y_pert = y_pred + epsilons * h

        # Targeted regularization term
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        # Total loss
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss