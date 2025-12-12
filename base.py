class BaseEstimator:
    """
    Base class for all estimators in the simple_ml library.
    Provides consistent interface for methods (fit, predict, etc)
    """

    def __init__(self, **kwargs):
        """
        Initialize the BaseEstimator with given parameters.

        :param kwargs: Hyperparameters set here
        """
        pass

    def fit(self, X, y):
        """
        Fit the model to the training data.
        Must be implemented by subclasses.

        :param X: Training data features
        :param y: Training data labels
        :return: self
        """
        raise NotImplementedError(
            "The `fit` method must be implemented by the subclass."
        )

    def predict(self, X):
        """
        Predict using the fitted model.
        Must be implemented by subclasses.

        :param X: Data features to predict
        :return: Predicted labels
        """
        raise NotImplementedError(
            "The `predict` method must be implemented by the subclass."
        )

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data w/ performance metric.
        (R^2, accuracy, RMSE, etc)
        Must be implemented by subclasses.

        :param X: Data features for evaluation
        :param y: True labels for evaluation
        :return: Evaluation metric (e.g., accuracy, RMSE)
        """
        raise NotImplementedError(
            "The `evaluate` method must be implemented by the subclass."
        )

    # Util method for printing model object
    def __repr__(self):
        return f"<{self.__class__.__name__}>"
