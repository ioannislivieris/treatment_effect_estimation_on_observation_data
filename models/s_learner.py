import numpy as np
from sklearn.ensemble import RandomForestClassifier


class S_Learner:
    """
    S-Learner for causal effect estimation.

    The S-Learner trains a single outcome model using both
    covariates and the treatment indicator as input features.

    Potential outcomes are estimated by toggling the treatment
    indicator at prediction time.
    """

    def __init__(self, base_learner=None, propensity_model=None):
        """
        Initialize the S-Learner.

        Parameters
        ----------
        base_learner : object
            An unfitted regression model that supports .fit() and .predict().
        propensity_model : object, optional
            A scikit-learn compatible classifier used to estimate
            propensity scores.
        """
        if base_learner is None:
            raise ValueError("A base_learner must be provided.")

        # Single outcome model trained on (X, T)
        self.model = base_learner

        # Initialize propensity score model (defaults to a balanced random forest)
        self.propensity_model = propensity_model or RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X:np.ndarray=None, T:np.ndarray=None, Y:np.ndarray=None):
        """
        Fit the S-Learner outcome model and the propensity model.

        Parameters
        ----------
        X : array-like
            Feature matrix of shape (n_samples, n_features).
        T : array-like
            Binary treatment indicator of shape (n_samples,).
        Y : array-like
            Outcome variable of shape (n_samples,).
        """

        # Augment feature matrix with treatment indicator
        XT = np.column_stack([X, T])

        # Train single outcome model
        self.model.fit(XT, Y)

        # Fit propensity model on the original feature space
        self.propensity_model.fit(X, T)
        print("S-Learner fitted successfully.")

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

        n = X.shape[0]

        # Create treatment indicators for counterfactual prediction
        T_0 = np.zeros(n)
        T_1 = np.ones(n)

        # Augment features for control and treated scenarios
        X_0 = np.column_stack([X, T_0])
        X_1 = np.column_stack([X, T_1])

        # Predict potential outcomes
        y_0_hat = self.model.predict(X_0)
        y_1_hat = self.model.predict(X_1)
        
        return {
            'y_0_hat': y_0_hat,  # Outcome if untreated
            'y_1_hat': y_1_hat,  # Outcome if treated
            'propensity_score': self.propensity_model.predict_proba(X)[:, 1]
        }
