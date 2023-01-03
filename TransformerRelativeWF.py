"""Transformer decoder class for autoregressive generation
of ground state wavefunction for any given 1D spin chain.

Allows one to break up spin chain into smaller fixed bit strings,
and sample from dictionary of all possible such bit strings,
thus allowing for greater flexibility in training larger systems,
or systems with lots of entanglement
E.g. instead of {0,1} at each pos, can have {00, 01, 10, 11}
and build state from these "words".

Update rule for weights is based on gradients of energy function
obtained from variational monte-carlo, as in https://arxiv.org/pdf/2002.02973.pdf
"""
from typing import Optional, Tuple, Callable, Union
import tensorflow as tf
import numpy as np
from relativemultiheadattention import MultiHeadAttentionRelative


def get_angles(pos: int, i: int, d_model: int) -> float:
    """Generates angles for positional encoding

    Args:
        pos: int, position in sequence
        i: int, index along embedding dimension
        d_model: int, embedding dimension
    Returns:
        float, argument of sinusoids in absolute positional encoding
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def generate_positional_encoding(position: int, d_model: int) -> np.ndarray:
    """Assigns a positional encoding to each element of embedding
    vector at each position in sequence, as described in https://arxiv.org/pdf/2002.02973.pdf.

    Args:
        position: int, position of token
        d_model: int, embedding dimension
    Returns:
        np.ndarray, positional encoding for embeddings
    """
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.cast(angle_rads, tf.float32)


def create_look_ahead_mask(size: int) -> tf.Tensor:
    """Generates causal mask (upper triangular matrix) for decoder,
    gets applied to attn weights.

    Args:
        size: int, sequence length
    Returns:
        tf.Tensor, triangular matrix that masks future tokens
    """
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def pad_states(states: tf.Tensor) -> tf.Tensor:
    """Pads start of sequence with zero to act as seed token for autoregressive generation.

    Args:
        states: tf.tensor, shape (batch_size, sequence_length)
    Returns:
        tf.Tensor, states pre-padded with seed token
    """
    return np.pad(states, (1, 0), mode="constant")[1:]


def spins_to_labels(vocab_length: int, spins: tf.Tensor) -> tf.Tensor:
    """Maps a set of spins to a set of spin "words"

    Args:
        vocab_length: int, total number of spin "words" in dictionary
                       all words have same length
        spins: tf.Tensor, binary sequence, shape (batch, length)
    Returns:
        tf.Tensor, converts spins into labels for each elem of dictionary
    """
    batch_size = spins.shape[0]
    length_vocab_elem = int(np.log2(vocab_length))

    assert (
        spins.shape[-1] % length_vocab_elem == 0
    ), "word size must divide length of spin chain"
    spins_length = spins.shape[-1] // length_vocab_elem

    # break up spin chain into spin "words"

    spins = tf.reshape(spins, (batch_size, spins_length, -1))

    # map from bit strings to labels

    return tf.tensordot(
        spins.astype("int32"),
        2 ** np.arange(spins.shape[-1])[::-1].astype("int32"),
        axes=1,
    )


def labels_to_spins(vocab_length: int, labels: tf.Tensor) -> tf.Tensor:
    """Inverse of spins_to_labels method.
    """
    batch_size = labels.shape[0]
    length_vocab_elem = int(np.log2(vocab_length))
    spin_vocab = []

    # generate dictionary from labels to bit strings

    for i in range(vocab_length):
        spin_vocab.append(list(np.binary_repr(i, width=length_vocab_elem)))

    # project onto relevant bit strings and glue them back together to get spin chain

    spins_from_labels = tf.gather(
        np.array(spin_vocab, dtype=int), labels.astype(int), axis=-1
    )
    return tf.reshape(spins_from_labels, (batch_size, -1))


class TransformerDecoder(tf.keras.Model):
    """Decoder which autoregressively generates wavefunction for a given Hamiltonian

    IMPORTANT: sequence length includes the start token, and indicates the
    length of the sequences used during training. This means that if we set
    sequence length to be 11, we'll be considering states with maximum length 10.
    Technically we dont need a fixed maximum sequence length, but this makes sense
    in the context of hamiltoninans.

    Dropout is applied just as in https://arxiv.org/pdf/1706.03762.pdf.

    Attributes:
        num_heads: int,
        key_dim: int,
        value_dim: int,
        embedding_dim: int,
        dictionary_size: int, The number of "words", ie all possible bit strings
            of a given size, from which sequence is generated
        decoding_reps: int, The number of repetitions of attention,
            followed by a feed-forward network.
        width: int, The width of the feed forward network if it is
            chosen to be fully-connected
        sequence_length: int, The sequence length used during training;
            this includes the start-token
        depth: int = 2, The depth of the feed-forward networks used
        max_distance: int, The cutoff distance for relative positional encoding
        random_positional_encoding: bool, Whether the positional encoding
            is taken to be random, or the version used in
            'attention is all you need'
        trainable_positional_encoding: bool, Whether the positional
            embedding is trainable
        conv_feed_forward: Optional[Tuple[int, int]] = (3, 1),
            either None, indicating we use a dense network, or
            a tuple giving us the parameters for our conv1d network:
            conv_feed_forward = (filters, kernel_size)
        name: str = None, The name of this model
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        embedding_dim: int,
        dictionary_size: int,
        decoding_reps: int,
        width: int,
        sequence_length: int,
        max_distance: int,
        depth: int = 2,
        final_temperature: float = 1,
        random_positional_encoding: bool = False,
        trainable_positional_encoding: bool = False,
        conv_feed_forward: Optional[Tuple[int, int]] = (3, 1),
        dropout: float = 0,
        attention_dropout: float = 0,
        name: str = None,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.embedding_dim = embedding_dim
        self.dictionary_size = dictionary_size
        self.decoding_reps = decoding_reps
        self.width = width
        self.depth = depth
        self.sequence_length = sequence_length
        self.max_distance = max_distance
        self.random_positional_encoding = random_positional_encoding
        self.conv_feed_forward = conv_feed_forward
        self.final_temperature = final_temperature
        self.droput_rates = (dropout, attention_dropout)
        self.trainable_positional_encoding = trainable_positional_encoding

        # define the functional model below

        if self.random_positional_encoding:
            self.positional_encodings = tf.Variable(
                tf.random.normal(shape=(self.sequence_length, self.embedding_dim)),
                trainable=self.trainable_positional_encoding,
            )
        else:
            self.positional_encodings = tf.Variable(
                generate_positional_encoding(self.sequence_length, self.embedding_dim),
                trainable=self.trainable_positional_encoding,
            )

        self.embeddings = tf.keras.layers.Embedding(
            input_dim=self.dictionary_size,
            output_dim=self.embedding_dim,
            trainable=True,
        )

        self.attention_layers = [
            MultiHeadAttentionRelative(
                num_heads=self.num_heads,
                head_size=self.key_dim,
                output_size=self.value_dim,
                max_distance=self.max_distance,
                dropout=self.dropout_rates[1],
                use_projection_bias=True,
                return_attn_coef=True,
                name=f"attention_{idx}",
            )
            for idx in range(self.decoding_reps)
        ]

        if conv_feed_forward is None:
            self.feed_forward = [
                [
                    tf.keras.layers.Dense(
                        self.width,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        name=f"fully-connected_{ff_idx}_layer_{layer_idx}",
                    )
                    for layer_idx in range(self.depth - 1)
                ]
                + [
                    tf.keras.layers.Dense(
                        self.embedding_dim,
                        activation=None,
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        name=f"fully-connected_{ff_idx}_layer_{self.depth - 1}",
                    )
                ]
                for ff_idx in range(self.decoding_reps)
            ]
        else:
            filters, kernel_size = self.conv_feed_forward

            self.feed_forward = [
                [
                    tf.keras.layers.Conv1D(
                        filters=filters,
                        kernel_size=kernel_size,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        name=f"fully-connected_{ff_idx}_layer_{layer_idx}",
                    )
                    for layer_idx in range(self.depth - 1)
                ]
                + [
                    tf.keras.layers.Dense(
                        embedding_dim,
                        activation=None,
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        name=f"fully-connected_{ff_idx}_layer_{self.depth - 1}",
                    )
                ]
                for ff_idx in range(self.decoding_reps)
            ]

        self.layer_norms = [
            [
                tf.keras.layers.LayerNormalization(axis=-1, trainable=False)
                for _ in range(2)
            ]
            for _ in range(self.decoding_reps)
        ]

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rates[0])

        # here we define a glorot normal initializer with std scaled by the temperature

        self.final_kernel_initializer = tf.keras.initializers.RandomNormal(
            mean=0.0,
            stddev=np.sqrt(2 / (self.embedding_dim + self.dictionary_size))
            / np.sqrt(self.final_temperature),
        )

        self.final_layer = tf.keras.layers.Dense(
            dictionary_size,
            activation=tf.nn.log_softmax,
            use_bias=False,
            kernel_initializer=self.final_kernel_initializer,  #'glorot_uniform',
            name="final_linear_layer",
        )

        self.phase_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            name="final_linear_layer",
        )

        self.build((None, self.sequence_length))

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Args:
            inputs, tf.Tensor, tensor of tokens with shape (batch, length)
        Returns:
            tf.Tensor, logits from forward pass
        """

        embeddings = self.embeddings(inputs)

        assert (
            embeddings.shape[-2] <= self.sequence_length
        ), f"expected sequence length <= {self.sequence_length}, recieved length {embeddings.shape[-2]}"

        argument = self.dropout_layer(
            embeddings + self.positional_encodings[: embeddings.shape[-2], :]
        )

        mask = create_look_ahead_mask(embeddings.shape[-2])

        attn_weights_layers = []

        for idx in range(self.decoding_reps):
            attention_output, attention_weights = self.attention_layers[idx](
                [argument, argument], mask=mask
            )

            attn_weights_layers.append(attention_weights)

            attention_output = self.dropout_layer(attention_output)

            normalized_residual_output = self.layer_norms[idx][0](
                argument + attention_output
            )

            argument = normalized_residual_output

            for feed_forward_layer in self.feed_forward[idx]:
                argument = feed_forward_layer(argument)

            self.dropout_layer(argument)

            argument = self.layer_norms[idx][1](argument + normalized_residual_output)

        logits = self.final_layer(argument)

        phase = np.pi * tf.nn.softsign(tf.squeeze(self.phase_layer(argument), axis=-1))

        return logits, phase, attn_weights_layers

    def autoregressive_sampling(
        self, initial_states: tf.Tensor
    ) -> Tuple[np.ndarray, tf.Tensor, tf.Tensor]:
        """Implements autoregressive sampling,
        to be run this within gradient tape.

        Cuts off the initial (seed) spin before returning.

        Args:
            initial_states: tf.Tensor, shape (batch_size, intial_length)
        Returns:
            tuple of tf.Tensor, the generated states, and associated
            logits and phases
        """
        batch_size = initial_states.shape[0]

        autoregressive_spin_states = np.copy(initial_states)

        autoregressive_phases = tf.zeros(batch_size)
        log_probabilities = tf.zeros(batch_size)

        for idx in range(self.sequence_length):
            conditional_logits, phases, _ = self.call(inputs=autoregressive_spin_states)

            # if we are past the start token, accumulate the phase

            if autoregressive_spin_states.shape[-1] > 1:
                autoregressive_phases += phases[:, -1]

            # if we aren't at the final sequence element, select dictionary items

            if idx < self.sequence_length - 1:
                choices = tf.random.categorical(
                    conditional_logits[:, -1, :], num_samples=1, dtype=tf.int32,
                )

                autoregressive_spin_states = np.hstack(
                    [autoregressive_spin_states, choices]
                )

                log_probabilities += tf.gather(
                    conditional_logits[:, -1, :],
                    tf.reshape(choices, -1),
                    batch_dims=1,
                    axis=-1,
                )
        return (
            np.array(autoregressive_spin_states)[..., 1:],
            log_probabilities,
            autoregressive_phases,
        )

    def evaluate_state(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Given a set of basis states, returns the logits and phases
        of the wavefunction in that basis.

        Args:
            state: tf.Tensor, shape (batch_size, sequence_length)
        Returns:
            tuple of tf.Tensor, the logits and phases
        """
        # if incoming state is bit string (as in Hamiltonian computation) convert to labels

        if state.shape[-1] > self.sequence_length - 1:
            state = spins_to_labels(spins=state, vocab_length=self.dictionary_size)
        state = state.astype(np.int32)

        conditional_logits, phases, _ = self.call(inputs=pad_states(state),)

        # collect the total phases (aside from the final phase)
        # and the conditional probabilities

        total_phases = tf.reduce_sum(phases[:, 1:], -1)

        log_probs = tf.reduce_sum(
            tf.gather(conditional_logits[:, :-1, :], state, batch_dims=2, axis=-1),
            axis=-1,
        )

        return log_probs, total_phases

    def evaluate_gradients(
        self,
        batch_size: int,
        local_energy_function: Callable,
        return_energy: bool = True,
        reps: int = 1,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, float, float]]:
        """Returns the gradients of the Hamiltonian wrt weights,
        using variational Monte-Carlo.

        This uses (C6) of https://arxiv.org/pdf/2002.02973.pdf.

        Args:
            batch_size: int, the number of samples
            local_energy_function: the Hamiltonian function
            return_energy: bool, whether to also return average energy
        Returns:
            tf.Tensor, gradients of loss
            with option to include floats, real/imag parts of
            average energy
        """

        initial_states = np.zeros((batch_size, 1))

        # generate samples

        all_samples = []
        e_estimate_real_parts = []
        e_estimate_imag_parts = []

        for _ in range(reps):
            samples, log_probs, phases = self.autoregressive_sampling(initial_states)
            all_samples.append(samples)

            # We dont want gradients passing through E_loc so we use their numpy counterparts
            # convert labels to spins for Hamiltonian computation

            e_loc_real_part, e_loc_imag_part = local_energy_function(
                self, labels_to_spins(labels=samples, vocab_length=self.dictionary_size)
            )

            if tf.is_tensor(e_loc_real_part):
                e_loc_real_part = e_loc_real_part.numpy()
            if tf.is_tensor(e_loc_imag_part):
                e_loc_imag_part = e_loc_imag_part.numpy()

            e_estimate_real_part = np.mean(e_loc_real_part)
            e_estimate_imag_part = np.mean(e_loc_imag_part)

            e_estimate_real_parts.append(e_estimate_real_part)
            e_estimate_imag_parts.append(e_estimate_imag_part)

        with tf.GradientTape() as tape:
            loss = 0

            # evaluate samples

            for sample_idx, samples in enumerate(all_samples):
                log_probs, phases = self.evaluate_state(samples,)

                # Construct the loss function
                # For each state this becomes (1/2)*D_log_p*Re[E_{loc} - E]+D_phase*Im[E_{loc} - E]

                e_estimate_real_part = e_estimate_real_parts[sample_idx]
                e_estimate_imag_part = e_estimate_imag_parts[sample_idx]

                loss += (2 / batch_size) * tf.reduce_sum(
                    (1 / 2) * log_probs * (e_loc_real_part - e_estimate_real_part)
                )

                loss += (2 / batch_size) * tf.reduce_sum(
                    phases * (e_loc_imag_part - e_estimate_imag_part)
                )

            loss /= reps

        derivatives = tape.gradient(loss, self.trainable_weights)

        if return_energy:
            return (
                derivatives,
                (np.mean(e_estimate_real_parts), np.mean(e_estimate_imag_parts)),
            )

        return derivatives
