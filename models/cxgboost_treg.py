import time
import xgboost
import numpy as np
from typing import Dict, Optional

from models.cxgboost import XGB_2D_GRADIENT


class CXGBoostTreg:
    """
    C-XGBoost with Targeted Regularization.

    Extends C-XGBoost by adding:
      - A third output head for propensity score (sigmoid of z)
      - Targeted regularization term in the loss function

    Loss (per sample):
        L = (1-t)(y-Q0)^2 + t(y-Q1)^2
          + alpha * CrossEntropy(g_hat, t)
          + beta  * [(1-t)(y-Q0_tilde)^2 + t(y-Q1_tilde)^2]

    where:
        Q0_tilde = Q0 - epsilon / (1 - g_hat)
        Q1_tilde = Q1 + epsilon / g_hat
        g_hat    = sigmoid(z)
    """

    def __init__(
        self,
        parameters: Optional[dict] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        epsilon: float = 0.01,
        verbose: bool = False
    ) -> None:
        """
        Args:
            parameters: XGBoost hyperparameters
            alpha:      Weight for propensity score cross-entropy term
            beta:       Weight for targeted regularization term
            epsilon:    Targeted regularization shift parameter
            verbose:    Flag for printing
        """
        self.parameters = parameters or {}
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.verbose = verbose

        self.model = xgboost.XGBRegressor(**self.parameters, verbosity=0)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return np.where(z >= 0,
                        1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    @staticmethod
    def _clip_propensity(g: np.ndarray, eta: float = 1e-6) -> np.ndarray:
        return np.clip(g, eta, 1.0 - eta)

    def _get_loss(self, t: np.ndarray):
        """Build and return the custom XGBoost objective."""
        alpha = self.alpha
        beta = self.beta
        epsilon = self.epsilon

        def custom_loss(y_true: np.ndarray, y_pred: np.ndarray):
            n = len(t)

            # y_pred is flattened row-major: [Q0_0, Q1_0, z_0, Q0_1, Q1_1, z_1, ...]
            pred = y_pred.reshape(n, 3)
            Q0 = pred[:, 0]
            Q1 = pred[:, 1]
            z  = pred[:, 2]

            y = y_true.reshape(n, 3)[:, 0]

            g_hat = self._clip_propensity(self._sigmoid(z))

            Q0_tilde = Q0 - epsilon / (1.0 - g_hat)
            Q1_tilde = Q1 + epsilon / g_hat

            # Gradients (Proposition 1 / Appendix B)
            grad_Q0 = np.where(t == 0, -2.0 * (y - Q0) - 2.0 * beta * (y - Q0_tilde), 0.0)
            grad_Q1 = np.where(t == 1, -2.0 * (y - Q1) - 2.0 * beta * (y - Q1_tilde), 0.0)

            # Targeted-regularisation residual r = y - Q_tilde and its
            # derivative w.r.t. z (dg/dz = g(1-g)):
            #   t=0: dr/dz = +eps * g / (1-g)   (= eps * e^z)
            #   t=1: dr/dz = +eps * (1-g) / g   (= eps * e^-z)
            # so d/dz [beta r^2] = 2 beta r dr/dz, and
            #   d2/dz2 [beta r^2] = 2 beta [(dr/dz)^2 + r d2r/dz2],
            # with d2r/dz2 = +dr/dz (t=0) and -dr/dz (t=1).
            r = np.where(t == 0, y - Q0_tilde, y - Q1_tilde)
            dr_dz = np.where(t == 0,
                             epsilon * g_hat / (1.0 - g_hat),
                             epsilon * (1.0 - g_hat) / g_hat)
            d2r_dz2 = np.where(t == 0, dr_dz, -dr_dz)

            grad_z = alpha * (g_hat - t) + 2.0 * beta * r * dr_dz

            # Hessians
            hess_Q0 = np.where(t == 0, 2.0 * (1.0 + beta), 1e-16)
            hess_Q1 = np.where(t == 1, 2.0 * (1.0 + beta), 1e-16)
            hess_z  = np.maximum(
                alpha * g_hat * (1.0 - g_hat)
                + 2.0 * beta * (dr_dz ** 2 + r * d2r_dz2),
                1e-16
            )

            # (n, 3) on xgboost >= 2.1, flattened on older versions.
            shape = (n, 3) if XGB_2D_GRADIENT else (-1,)
            grad = np.column_stack([grad_Q0, grad_Q1, grad_z]).reshape(shape)
            hess = np.column_stack([hess_Q0, hess_Q1, hess_z]).reshape(shape)

            return grad, hess

        return custom_loss

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model.

        Args:
            X: Covariates,          shape (n, d)
            t: Treatment indicator, shape (n,), binary 0/1
            y: Observed outcome,    shape (n,)
        """
        custom_loss = self._get_loss(t)

        self.model.set_params(
            objective=custom_loss,
            num_target=3,
            # multi_strategy="multi_output_tree",
        )

        yt = np.column_stack([y, y, y])

        start = time.time()
        if self.verbose:
            print("[INFO] Fitting C-XGBoost (treg)...")
        self.model.fit(X, yt)
        if self.verbose:
            print(f"[INFO] Training completed in {time.time() - start:.2f}s")

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict potential outcomes and propensity score.

        Args:
            X: Covariates, shape (n, d)

        Returns:
            Dictionary with:
                - 'y0_hat':           Predicted outcome under control (T=0)
                - 'y1_hat':           Predicted outcome under treatment (T=1)
                - 'propensity_score': P(T=1|X) = sigmoid(z)
        """
        pred = self.model.predict(X)   # shape (n, 3)

        return {
            "y_0_hat":           pred[:, 0],
            "y_1_hat":           pred[:, 1],
            "propensity_score": self._clip_propensity(self._sigmoid(pred[:, 2])),
        }