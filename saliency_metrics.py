"""Collection of saliency metrics for trained decoder:
-input x gradient
-(integrated) attention x gradient
-spin flip
Spin flip saliency = feed-forward perturbation,
while the other two are gradient saliency metrics.

All saliency metrics below measure dependence of
logits at a given position with tensors at other
positions, e.g. input embedding weights or attention
weights.
"""
import copy
import numpy as np
import tensorflow as tf
from TransformerRelativeWF import TransformerDecoder


def gradient_saliency(decoder: TransformerDecoder, samples: tf.Tensor) -> tf.Tensor:
    """Computes input x gradient saliency s(i,j) for tokens i,j according to
    s(i,j) = Xi * nabla_Xi Fj, where Xi is the input vector at i
    and Fj is the logits distribution at j

    See https://jalammar.github.io/explaining-transformers/

    Args:
        decoder: TransformerDecoder, the transformer wavefunction
        samples: tf.Tensor, basis states for sampling
    Returns:
        tf.Tensor, saliency tensor
    """

    batch_size, seq_length = samples.shape

    saliency_tensor = np.zeros((batch_size, seq_length, seq_length))

    embeddings = decoder.embeddings(samples)

    # Important point: the input vectors are discrete so in computing saliency we actually
    # need to take Xi to be the embedding vector at i
    # But we can't directly differentiate wrt embedding vector, since
    # it is output of an intermediate layer

    # Instead we differentiate wrt embedding weights;
    # tensorflow automatically slices this according to input vector at each i,
    # effectively giving us gradients wrt embedding vectors at i

    with tf.GradientTape(persistent=True) as tape:

        logits, _, _ = decoder(samples)

        # get max logit for each site

        max_logits = tf.gather(
            tf.reshape(logits, (batch_size * seq_length, -1)),
            tf.reshape(tf.argmax(logits, axis=-1), (batch_size * seq_length,)),
            axis=-1,
            batch_dims=1,
        )
        max_logits = tf.reshape(max_logits, (batch_size, seq_length))

        all_max_logits = []

        # collapse max_logits into a single array so that gradient tape doesn't
        # automatically average over batch (could also use batch jacobian)

        for batch_idx in range(batch_size):
            for idx in range(seq_length):
                all_max_logits.append(max_logits[batch_idx, idx])

    for batch_idx in range(batch_size):
        for idx in range(seq_length - 1):
            grads = tape.gradient(
                all_max_logits[batch_idx * seq_length + idx + 1],
                decoder.embeddings.weights[0],
            ).values[: idx + 1]

            saliency_tensor[batch_idx, : idx + 1, idx + 1] = tf.norm(
                grads * embeddings[batch_idx, : idx + 1, :], axis=-1
            )

    return saliency_tensor


def gradient_attention_score(
    decoder: TransformerDecoder, samples: tf.Tensor, riemann_steps: int = 1
) -> tf.Tensor:
    """Computes an integrated attention x gradient saliency.

    We define this as the combination of
    eq (2) of https://arxiv.org/abs/2204.11073
    and eq (4) of https://arxiv.org/pdf/2004.11207.pdf

    Explicitly, we define the saliency by
    s(i,j) = aij * int_0^1 dc ReLU(nabla_aij F(c*aij))
             ~ aij * sum_m ReLU(nabla_aij F(m/steps aij))

    Args:
        decoder: TransformerDecoder, the transformer wavefunction
        riemann_steps: int, number of discrete steps in Riemann sum to approx
                        integral
    Returns:
        tf.Tensor, integrated attention x gradient saliency
    """

    batch_size, seq_length = samples.shape
    num_heads = decoder.num_heads
    num_layers = decoder.decoding_reps

    attention_saliency_tensor = np.zeros(
        (num_layers, batch_size, num_heads, seq_length, seq_length)
    )

    orig_scaling = [
        tf.Variable(decoder.layers[idx].weights[-2]) for idx in range(1, num_layers + 1)
    ]

    assert (
        decoder.for_saliency is True
    ), "Decoder must be instantiated with for_saliency = True"

    with tf.GradientTape(persistent=True) as tape:

        for idx in range(1, num_layers + 1):

            # fiducial bias is not trainable so manually track it on tape

            tape.watch(decoder.layers[idx].weights[-1])

        all_max_logits = [[] for _ in range(riemann_steps)]

        for m in range(1, riemann_steps + 1):

            rescaling = m / riemann_steps

            for layer_idx in range(1, num_layers + 1):

                # generate one-param family of rescalings

                decoder.layers[layer_idx].weights[-2].assign(
                    rescaling * orig_scaling[layer_idx - 1]
                )

            logits, _, attention_weights = decoder(samples)

            # get max logit for each site

            max_logits = tf.gather(
                tf.reshape(logits, (batch_size * seq_length, -1)),
                tf.reshape(tf.argmax(logits, axis=-1), (batch_size * seq_length,)),
                axis=-1,
                batch_dims=1,
            )
            max_logits = tf.reshape(max_logits, (batch_size, seq_length))

            for batch_idx in range(batch_size):
                for idx in range(seq_length):
                    all_max_logits[m - 1].append(max_logits[batch_idx, idx])

    # compute integrated saliency by approx as Riemann sum

    for layer_idx in range(1, num_layers + 1):
        for batch_idx in range(batch_size):
            for idx in range(seq_length - 1):
                for m in range(1, riemann_steps + 1):
                    grads = tape.gradient(
                        all_max_logits[m - 1][batch_idx * seq_length + idx + 1],
                        decoder.layers[layer_idx].weights[-1],
                    )[batch_idx, :, idx + 1, : idx + 1]

                    attention_saliency_tensor[
                        layer_idx - 1, batch_idx, :, idx + 1, : idx + 1
                    ] += (
                        tf.nn.relu(grads)
                        * attention_weights[layer_idx - 1][
                            batch_idx, :, idx + 1, : idx + 1
                        ]
                    )

    return np.mean(attention_saliency_tensor, axis=(0, 1, 2)) / riemann_steps


def spin_flip(samples: tf.Tensor, pos: int) -> tf.Tensor:
    """Flips spin at given position.
    """

    flipped_samples = copy.deepcopy(samples)
    flipped_samples[:, pos] = (-1 * (2 * samples[:, pos] - 1) + 1) / 2

    return flipped_samples


def kl_divergence(logits_a: tf.Tensor, logits_b: tf.Tensor) -> tf.Tensor:
    """Computes KL divergence.
    """
    probs_a = tf.nn.softmax(logits_a, axis=-1)
    probs_b = tf.nn.softmax(logits_b, axis=-1)

    return np.einsum("ij, ij -> i", probs_a, np.log(probs_a / probs_b))


def js_divergence(logits_a: tf.Tensor, logits_b: tf.Tensor) -> tf.Tensor:
    """Computes JS divergence.
    """
    avg_logits = 1 / 2 * (logits_a + logits_b)
    return (
        1
        / 2
        * (kl_divergence(logits_a, avg_logits) + kl_divergence(logits_b, avg_logits))
    )


def spin_flip_saliency(decoder: tf.Tensor, samples: tf.Tensor) -> tf.Tensor:
    """Forward pass perturbation saliency: we flip input spin at some site
    and see how it changes conditional probabilities at a later site.

    New and old probabilitiy distributions at later site are
    compared with JS divergence.

    Args:
        decoder: TransformerDecoder, the transformer wavefunction
        samples: tf.Tensor, states for sampling
    Returns:
        tf.Tensor, spin flip saliency as described above
    """

    batch_size, seq_length = samples.shape

    logits, _, _ = decoder(samples)

    saliency_tensor = np.zeros((batch_size, seq_length, seq_length))

    for idx in range(seq_length):
        for idy in range(idx):
            flipped_samples = spin_flip(samples, idy)
            flipped_logits, _, _ = decoder(flipped_samples)
            saliency_tensor[:, idx, idy] = js_divergence(
                flipped_logits[:, idx, :], logits[:, idx, :]
            )

    return saliency_tensor
