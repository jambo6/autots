from sklearn.base import TransformerMixin


class NoFitTransformerMixin(TransformerMixin):
    """ Transformer mixin with a fit method that returns self. """

    def fit(self, data, labels=None):
        self._is_fitted = True
        return self
