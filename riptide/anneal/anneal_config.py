class Config(object):
    """Configuration scope of current mode.

    This is used to easily switch between different
    model structure variants by simply calling into these functions.

    Parameters
    ----------
    quantize: bool
        whether to apply quantization or not.
    w_bits : Tensor
        number of weight bits to use
    a_bits : Tensor
        number of activation bits to use
    fixed : Whether to use learned clipping or fixed 1.0
    """
    current = None

    def __init__(self, quantize=False, a_bits=None, w_bits=None, fixed=True):
        self.quantize = quantize
        self.a_bits = a_bits
        self.w_bits = w_bits
        self.fixed = fixed

    def __enter__(self):
        self._old_manager = Config.current
        Config.current = self
        return self

    def __exit__(self, ptype, value, trace):
        Config.current = self._old_manager
