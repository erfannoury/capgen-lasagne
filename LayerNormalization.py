import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import Layer

__all__ = [
    "LayerNormalizationLayer"
]

class LayerNormalizationLayer(Layer):
    """
    LayerNormalizationLayer(incoming, feat_dims,
    alpha=lasagne.init.Constant(1.0), beta=lasagne.init.Constant(0.0),
    **kwargs)

    A Layer Normalization layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    feat_dims : int
        The number of feature dimensions. The last dimensions for
        features of each instance will be normalized.

    alpha : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the alpha.
        This should be a tensor with shape ``(feat1, feat2, ...)``.
        See :func:`lasagne.utils.create_param` for more information.

    beta : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the beta.
        This should be a vector with shape ``(feat1, feat2, ...)``.
        See :func:`lasagne.utils.create_param` for more information.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> from LayerNormalization import LayerNormalizationLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)
    >>> ln = LayerNormalizationLayer(l1, 1)

    """
    def __init__(self, incoming, feat_dims, 
                 alpha=lasagne.init.Constant(1.0),
                 beta=lasagne.init.Constant(0.0),
                 eps = 1e-5,
                 **kwargs):
        super(LayerNormalizationLayer, self).__init__(incoming, **kwargs)

        if feat_dims < 1:
            raise ValueError("Number of feature dimensions should be"
                             "greater than zero.")
        self.feat_dims = feat_dims
        self.eps = eps

        self.feat_shape = self.input_shape[-self.feat_dims:]

        self.norm_axes = tuple([-i for i in xrange(1, feat_dims + 1)][::-1])

        self.alpha = self.add_param(alpha, self.feat_shape, name="alpha")
        self.beta = self.add_param(beta, self.feat_shape, name="beta",
                                   regularizable=False)

    def get_output_shape_for(self, input_shape):
        return self.input_shape

    def get_output_for(self, input, **kwargs):
        
        input = (input - input.mean(self.norm_axes, keepdims=True)) / \
                T.sqrt(input.var(self.norm_axes, keepdims=True) + self.eps)
        
        alpha = T.shape_padleft(self.alpha, input.ndim - self.feat_dims)

        return input * alpha + self.beta