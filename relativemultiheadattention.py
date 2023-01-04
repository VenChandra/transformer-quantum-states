"""Custom multi-head attention layer with relative positional encodings,
https://arxiv.org/abs/1803.02155,
as well as fiducial non-trainable weights for (integrated) attention gradient saliency.

Relative positional encodings are a set of edge (link) weights
which augment absolute positional encoding
by features dependent on relative distances between tokens.
Additively modifies attention weights and value according to eqs (3) and (4) in paper.
"""

from typing import Optional, Union, Tuple
import tensorflow as tf


def generate_relative_positions_matrix(length: int, max_distance: int) -> tf.Tensor:
    """Generates matrix of relative positions between tokens,
    with clipping

    Following
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    Args:
        length: int, length of sequence
        max_distance: int, cutoff for relative distances
    Returns:
        tf.Tensor, shape = (length, length)
    """
    range_vec = tf.range(length)
    range_matrix = tf.reshape(tf.tile(range_vec, [length]), [length, length])

    dist_matrix = range_matrix - tf.reshape(range_vec, [length, 1])
    clipped_dist = tf.clip_by_value(dist_matrix, -max_distance, max_distance)

    shifted_dist = (
        clipped_dist + max_distance
    )  # shift relative distances to non-negative values

    return shifted_dist


def query_reshape_mult(query: tf.Tensor, key_edge: tf.Tensor) -> tf.Tensor:
    """Vectorized computation of query*key_edge in eq (4) of
    https://arxiv.org/abs/1803.02155

    Args:
        query: tf.Tensor, query tensor; shape (batch, length, heads, query_features)
        key_edge: tf.Tensor, key edge weights; shape (length, length, key_features)
    Returns:
        tf.Tensor

    einsum indices:
    b = batch
    l, L = length
    B = batch*heads
    d = depth
    """
    length, depth = key_edge.shape[0], key_edge.shape[-1]
    heads = query.shape[-2]

    query = tf.reshape(tf.einsum("blh... -> bhl...", query), shape=[-1, length, depth],)
    query = tf.einsum("Bl... -> lB...", query)

    output = tf.reshape(
        tf.einsum("lBd, lLd -> BlL", query, key_edge), [-1, heads, length, length],
    )

    return output


def attn_reshape_mult(attn: tf.Tensor, value_edge: tf.Tensor) -> tf.Tensor:
    """Vectorized computation of attn*value_edge in eq (3) of
    https://arxiv.org/abs/1803.02155

    Args:
        atnn: tf.Tensor, attention weights; shape (batch, heads, length, length)
        value_edge: tf.Tensor, value edge weights; shape (length, length, value_features)
    Returns:
        tf.Tensor
    """
    length, depth = value_edge.shape[0], value_edge.shape[-1]
    heads = attn.shape[1]

    attn = tf.reshape(attn, shape=[-1, length, length])
    attn = tf.einsum("BlL -> lBL", attn)

    output = tf.reshape(
        tf.einsum("lBL, lLd -> Bld", attn, value_edge),
        shape=[-1, heads, length, depth],
    )
    output = tf.einsum("bhl... -> blh...", output)

    return output


class MultiHeadAttentionRelative(tf.keras.layers.Layer):
    """Layer corresponding to multi-head attention
    with relative positional encoding,
    including fiducial weights.

    Partly following source code of
    https://www.tensorflow.org/addons/api_docs/python/tfa/layers/MultiHeadAttention

    Attributes:
        head_size: int, number of features for each head
        num_heads: int, number of heads
        max_distance: int, clipping distance
        output_size: int, feature dimension of layer output
        dropout: int, rate for dropout layers
        use_projection_bias: bool, whether to include bias for layer output
        return_attn_coef: bool, whether to return attention weights
        kernel_initializer: str, specifies initialization of weights
        bias_initializer: str, specifies initialization of bias
        for_saliency: bool, whether to build fiducial weights for saliency computation
        batch_size: int, only needs to be specified when computing attention saliency
    """

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        max_distance: int,
        output_size: int = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        for_saliency: bool = False,
        batch_size: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.head_size = head_size
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.output_size = output_size
        self.dropout = tf.keras.layers.Dropout(dropout)
        self._dropout_rate = dropout
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef
        self.for_saliency = (
            for_saliency  # toggles fiducial weights for saliency computation
        )
        self.batch_size = batch_size  # only relevant for attention saliency computation

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def build(self, input_shape: tf.Tensor):

        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        # for instantiating fiducial weights

        if self.for_saliency:
            num_key_elements = input_shape[1][-2]
            num_query_elements = input_shape[0][-2]

        num_pos_labels = 2 * self.max_distance + 1

        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_heads, num_query_features, self.head_size],
            initializer=self.kernel_initializer,
        )

        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=[self.num_heads, num_key_features, self.head_size],
            initializer=self.kernel_initializer,
        )

        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=[self.num_heads, num_value_features, self.head_size],
            initializer=self.kernel_initializer,
        )

        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=[self.num_heads, output_size, self.head_size],
            initializer=self.kernel_initializer,
        )

        self.embedding_key = self.add_weight(
            name="relative_key",
            shape=[num_pos_labels, num_key_features],
            initializer=self.kernel_initializer,
        )

        self.embedding_value = self.add_weight(
            name="relative_value",
            shape=[num_pos_labels, num_value_features],
            initializer=self.kernel_initializer,
        )

        if self.use_projection_bias:

            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=[output_size],
                initializer=self.bias_initializer,
            )
        else:
            self.projection_bias = None

        if self.for_saliency:

            # for generating one-parameter family of rescaled attn weights
            # when computing integrated saliency

            self.rescaling_attn = self.add_weight(
                name="rescaling_attn", shape=[1], initializer="ones", trainable=False
            )

            # attn -> attn + b so we can compute  df/dattn as df(attn + b)/db|_b = 0
            # f = model output, b = fiducial_bias

            self.fiducial_bias = self.add_weight(
                name="fiducial_bias",
                shape=[
                    self.batch_size,
                    self.num_heads,
                    num_query_elements,
                    num_key_elements,
                ],
                trainable=False,
                initializer=self.bias_initializer,
            )

    def call(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: Optional[tf.Tensor] = None,
    ) -> Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:

        """Performs forward pass through multihead attention layer,
        optionally returns attention weights.

        einsum indices:
        i = input features
        o = output features
        q = query elements
        k = key elements
        h = heads
        """

        query, key = inputs[0], inputs[1]
        value = inputs[2] if len(inputs) > 2 else key
        seq_length = query.shape[-2]

        query = tf.einsum("...qi,...hio -> ...qho", query, self.query_kernel)
        key = tf.einsum("...ki,...hio -> ...kho", key, self.key_kernel)
        value = tf.einsum("...ki,...hio -> ...kho", value, self.value_kernel)

        depth = tf.constant(self.head_size, dtype=query.dtype)  # casts to float
        query /= tf.sqrt(depth)

        relative_pos = generate_relative_positions_matrix(seq_length, self.max_distance)

        # maps relative positions to embedding space via lookup from weights

        key_edge = tf.gather(self.embedding_key, relative_pos, axis=0)
        value_edge = tf.gather(self.embedding_value, relative_pos, axis=0)

        logits = tf.einsum("...qhi,...khi -> ...hqk", query, key) + query_reshape_mult(
            query, key_edge,
        )

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            logits += -10e9 * (1 - mask)

        attn_coef = tf.nn.softmax(logits, axis=-1)
        attn_coef = self.dropout(attn_coef, training=training)

        if self.for_saliency:
            attn_coef = (
                attn_coef * self.rescaling_attn
            )  # rescale for integrated gradient
            attn_coef += self.fiducial_bias[
                ..., : attn_coef.shape[-2], : attn_coef.shape[-1]
            ]  # add ficudial weights to calculate derivative

        multihead_output = tf.einsum(
            "...hqk,...khi -> ...qhi", attn_coef, value
        ) + attn_reshape_mult(attn_coef, value_edge)

        output = tf.einsum(
            "...qhi,...hio -> ...qo", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        if self.return_attn_coef:
            return output, attn_coef
        return output
