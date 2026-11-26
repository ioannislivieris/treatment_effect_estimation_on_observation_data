from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

class T_Learner:
    """
    T-Learner for causal effect estimation.

    The T-Learner trains two separate outcome models:
    - One model on the control group (T = 0)
    - One model on the treated group (T = 1)

    A separate propensity model is used to estimate treatment assignment
    probabilities.
    """

    def __init__(self, base_learner=None, propensity_model=None):
        """
        Initialize the T-Learner.

        Parameters
        ----------
        base_learner : object
            An unfitted scikit-learn compatible regression model. Two
            independent copies (sklearn.base.clone) are fitted, one on the
            control group and one on the treated group.
        propensity_model : object
            A scikit-learn compatible classifier used to estimate
            propensity scores.
        """
        if base_learner is None:
            raise ValueError("A base_learner must be provided.")
        
        # Two independent copies: sharing one estimator object would make the
        # second fit overwrite the first, so that y_0_hat == y_1_hat.
        # Outcome model trained on control group (T = 0)
        self.model_0 = clone(base_learner)
        # Outcome model trained on treated group (T = 1)
        self.model_1 = clone(base_learner)

        # Initialize propensity score model (defaults to a balanced random forest)
        self.propensity_model = propensity_model or RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )
        
    def fit(self, X, T, Y):
        """
        Fit the T-Learner models and the propensity model.

        Parameters
        ----------
        X : array-like
            Feature matrix of shape (n_samples, n_features).
        T : array-like
            Binary treatment indicator of shape (n_samples,).
        Y : array-like
            Outcome variable of shape (n_samples,).
        """

        # Select samples from the control group
        X_0 = X[T == 0]
        Y_0 = Y[T == 0]
        # Train outcome model for control group
        self.model_0.fit(X_0, Y_0)
        
        # Select samples from the treated group
        X_1 = X[T == 1]
        Y_1 = Y[T == 1]
        # Train outcome model for treated group
        self.model_1.fit(X_1, Y_1)
    
        # Fit propensity model on the full dataset
        self.propensity_model.fit(X, T)
        
    def predict(self, X):
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
            - 'propensity_score': Estimated probability of control assignment.
        """

        # Generate structured causal predictions
        return {
            'y_0_hat': self.model_0.predict(X),  # Outcome if untreated
            'y_1_hat': self.model_1.predict(X),  # Outcome if treated
            'propensity_score': self.propensity_model.predict_proba(X)[:, 1]
        }
