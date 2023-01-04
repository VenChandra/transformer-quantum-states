"""Module for training transformer decoder
given a Hamiltonian.

We supply TFIM and MHS Hamiltonians
Hyperparameters are entered via terminal.
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow.python.ops.numpy_ops import np_config
from TransformerRelativeWF import TransformerDecoder

np_config.enable_numpy_behavior()

# check for a GPU

print(tf.config.list_physical_devices("GPU"))

# user enters hyperparams at train time
# NOTE: launch script from terminal with
# the cmd line arguments below

for idx, param in enumerate(sys.argv):
    if param in ("--system", "-s"):
        system_size = int(sys.argv[idx + 1])
    if param in ("--dict", "-d"):
        dictionary_size = int(sys.argv[idx + 1])
    if param in ("--heads", "-h"):
        num_heads = int(sys.argv[idx + 1])
    if param in ("--reps", "-r"):
        decoding_reps = int(sys.argv[idx + 1])
    if param in ("--warm", "-w"):
        warmup_steps = int(sys.argv[idx + 1])
    if param in ("--seed", "-e"):
        random_seed = int(sys.argv[idx + 1])
    if param in ("--clip", "-c"):
        max_distance = int(sys.argv[idx + 1])

np.random.seed(random_seed)

# critical point tfim Hamiltonian


def tfim_e_loc(
    configuration: tf.Tensor,
    decoder: TransformerDecoder,
    J: float = -1,
    g: float = -1,
    use_stoquastic: bool = True,
) -> tf.Tensor:
    """Computes TFIM Hamiltonian
    Args:
        configuration: tf.Tensor, spin states,
                       shape (batch, config_length),
                       config_length changes over training epochs
        decoder: TransformerDecoder, the transformer wavefunction
        J, g: int, the TFIM couplings
        use_stoquastic: bool, flag to indicate whether to freeze phases
    Returns:
        tf.Tensor, Hamiltonian of TFIM
    """
    batch_size = configuration.shape[0]
    config_length = configuration.shape[1]
    flip_off_diags = np.ones((config_length, config_length)) - 2 * np.identity(
        config_length
    )

    flipped_states = np.einsum("bl,lh->bhl", 2 * configuration - 1, flip_off_diags)
    flipped_states = ((flipped_states + 1) / 2).astype(np.int32)

    # include z contributions

    signs = (2 * configuration - 1).astype(np.int32)
    e_loc_real_part = J * np.sum(signs[..., 1:] * signs[..., :-1], -1)
    e_loc_imag_part = 0.0

    # include off-diagonal contributions
    # flipped_states has shape (batch_size, length, length)

    reshaped_flipped_states = np.reshape(flipped_states, (-1, config_length))

    log_probs, total_phases = decoder.evaluate_state(configuration,)

    flipped_log_probs, flipped_total_phases = decoder.evaluate_state(
        reshaped_flipped_states,
    )

    flipped_total_phases = tf.reshape(flipped_total_phases, (batch_size, -1))
    flipped_log_probs = tf.reshape(flipped_log_probs, (batch_size, -1))

    ratio_mags = tf.math.exp((1 / 2) * (flipped_log_probs - log_probs[..., None]))

    # Below, we effectively assert that phases for all states are 0

    if use_stoquastic:
        ratio_real_part = ratio_mags
        ratio_imag_part = tf.zeros_like(ratio_mags)
    else:
        ratio_real_part = ratio_mags * tf.math.cos(
            -total_phases[..., None] + flipped_total_phases
        )
        ratio_imag_part = ratio_mags * tf.math.sin(
            -total_phases[..., None] + flipped_total_phases
        )

    e_loc_real_part += g * tf.reduce_sum(ratio_real_part, -1)
    e_loc_imag_part = g * tf.reduce_sum(ratio_imag_part, -1)

    assert e_loc_real_part.shape[0] == batch_size
    assert e_loc_imag_part.shape[0] == batch_size

    return e_loc_real_part, e_loc_imag_part


# mhs Hamiltonian and helper methods


def inverse_d_squared(i: int, j: int, N: int) -> float:
    """Computes d in eq (8) of
    https://arxiv.org/pdf/1701.04844.pdf

    Args:
        i,j: int, positions in sequence
        N: int, total length of sequence
    Returns:
        float, coupling
    """
    mask = 1 * ((i - j) != 0)

    argument = np.pi * (j - i) / N

    d_value = N * np.abs(np.sin(argument)) / np.pi

    outputs = mask / (d_value ** 2 + (1 - mask))

    return outputs


def give_flip_matrix(i: int, config_length: int) -> np.ndarray:
    """Returns matrix with shape (length-1, length) that has
    the ith element flipped.

    Args:
        i: int, where to flip
        config_length: int, length of sequence
    Returns:
        np.ndarray, flipped matrix
    """
    flip_matrix = np.array(
        [
            [-1 if k in (i, j) else 1 for k in range(config_length)]
            for j in range(config_length)
            if j != i
        ]
    )
    return flip_matrix


def give_all_flips_i(configuration: np.ndarray, i: int,) -> np.ndarray:
    """Given states of shape (batch_size, length), return the states with
    the ith, jth spins flipped.
    """
    config_length = configuration.shape[1]
    flip_matrix = give_flip_matrix(i, config_length)

    # note this has shape (length-1, length)

    flipped_states = np.einsum("bl,hl->bhl", 2 * configuration - 1, flip_matrix)
    flipped_states = ((flipped_states + 1) / 2).astype(np.int32)

    return flipped_states


def mhs_e_loc(configuration: tf.Tensor, decoder: TransformerDecoder) -> tf.Tensor:
    """Computes Hamiltonian of modified Haldane-Shastry model,
    as in https://arxiv.org/pdf/1701.04844.pdf.
    """
    log_probs, total_phases = decoder.evaluate_state(configuration,)

    batch_size = configuration.shape[0]
    config_length = configuration.shape[1]

    # first we include z contributions

    inverse_d_squared_array = inverse_d_squared(
        np.arange(config_length)[:, None],
        np.arange(config_length)[None, :],
        N=config_length,
    )

    # divide by 2 for double counting

    d2 = (
        np.einsum(
            "ij,bi,bj->b",
            inverse_d_squared_array,
            1 - 2 * configuration,
            1 - 2 * configuration,
        )
        / 2
    )

    e_loc_real_part = d2
    e_loc_imag_part = 0

    # Now include off-diagonal contributions
    # To be safe, and to allow for large system sizes without
    # overwhelming memory, we go through this iteratively

    for i in range(config_length):

        # First compute all flipped states with i and
        # a different index flipped.
        # then evaluate the wavefunction for these states

        flipped_states = give_all_flips_i(configuration=configuration, i=i,)

        reshaped_flipped_states = np.reshape(flipped_states, (-1, config_length))

        flipped_log_prob_batches, flipped_total_phases_batches = [], []

        flipped_batch_size = reshaped_flipped_states.shape[0]

        for batch_idx in range(
            int(reshaped_flipped_states.shape[0] // flipped_batch_size) + 1
        ):

            flipped_log_probs, flipped_total_phases = decoder.evaluate_state(
                reshaped_flipped_states[
                    batch_idx
                    * flipped_batch_size : (batch_idx + 1)
                    * flipped_batch_size
                ],
            )

            flipped_total_phases_batches.append(flipped_total_phases.numpy())
            flipped_log_prob_batches.append(flipped_log_probs.numpy())

        flipped_total_phases = tf.concat(flipped_total_phases_batches, 0)
        flipped_log_probs = tf.concat(flipped_log_prob_batches, 0)

        flipped_total_phases = tf.reshape(flipped_total_phases, (batch_size, -1))
        flipped_log_probs = tf.reshape(flipped_log_probs, (batch_size, -1))

        # The coefficient here is (s_i*s_j - 1)
        # multiply each flipped contribution by this coefficient

        spins = 1 - 2 * configuration

        sj_values = np.hstack([spins[:, :i], spins[:, i + 1 :]])
        si_values = spins[:, i]

        d_values = np.array(
            [
                inverse_d_squared(i, j, N=config_length)
                for j in range(config_length)
                if j != i
            ]
        )

        current_coefficients = (si_values[:, None] * sj_values - 1) * d_values[None, :]

        ratio_mags = tf.math.exp((1 / 2) * (flipped_log_probs - log_probs[..., None]))

        ratio_real_part = ratio_mags * np.cos(
            -total_phases[..., None] + flipped_total_phases
        )
        ratio_imag_part = ratio_mags * np.sin(
            -total_phases[..., None] + flipped_total_phases
        )

        # again dividing by 2 to avoid double counting

        e_loc_real_part += (
            np.einsum("bj,bj->b", ratio_real_part, current_coefficients,) / 2
        )
        e_loc_imag_part += (
            np.einsum("bj,bj->b", ratio_imag_part, current_coefficients,) / 2
        )

    assert e_loc_real_part.shape[0] == batch_size
    assert e_loc_imag_part.shape[0] == batch_size

    return e_loc_real_part, e_loc_imag_part


# standard power lay decay for learning rate
# as in https://arxiv.org/pdf/1706.03762.pdf


class PolynomialSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Polynomial decay learning rate scheduler.

    Attributes:
        d_model: int, embedding dimension
        warmup: int, number of warmup steps
    """

    def __init__(self, d_model: int, warmup: int = 2500):
        super().__init__()

        self.d_model = d_model
        self.warmup = warmup

    def __call__(self, step: int) -> float:

        if step < self.warmup:
            return (1 / np.sqrt(self.d_model)) * step * 1 / np.power(self.warmup, 1.5)
        return (1 / np.sqrt(self.d_model)) * 1 / np.sqrt(step)


# optimal tuning of embedding dim with vocab size
# according to cite wiki page

embedding_dimension = 14 + int(np.log2(dictionary_size))

# sequence length is system size + 1

sequence_length = int(system_size // np.log2(dictionary_size)) + 1

model_name = (
    f"tfim_size_{system_size}_vocab_{dictionary_size}_heads_{num_heads}"
    f"_reps_{decoding_reps}_warmup_{warmup_steps}_clip_{max_distance}_seed_{random_seed}"
)

model_params = {
    "num_heads": num_heads,
    "key_dim": embedding_dimension,
    "value_dim": embedding_dimension,
    "embedding_dim": embedding_dimension,
    "dictionary_size": dictionary_size,
    "decoding_reps": decoding_reps,
    "max_distance": max_distance,
    "width": 15,
    "depth": 2,
    "sequence_length": sequence_length,
    "dropout": 0,
    "attention_dropout": 0,
    "final_temperature": 1,
}

# save decoder params for easy model loading in analysis modules

with open(f"params_{model_name}.json", "w") as outfile:
    json.dump(model_params, outfile)

decoder_train = TransformerDecoder(**model_params)

learning_rate = PolynomialSchedule(d_model=embedding_dimension, warmup=warmup_steps,)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

real_energies = []
imag_energies = []

NUM_EPOCHS = 1_000
BATCH_SIZE = 100

# choose Hamiltonian for training
# tfim is written as default but we provide
# and use both tfim and mhs Hamiltonians

for idx in range(NUM_EPOCHS):

    derivatives, (e_real, e_imag) = decoder_train.evaluate_gradients(
        BATCH_SIZE, tfim_e_loc
    )

    optimizer.apply_gradients(zip(derivatives, decoder_train.trainable_weights))

    real_energies.append(e_real)
    imag_energies.append(e_imag)

    if idx % 50 == 0 and idx > 0:
        print(f"step {idx-5} to {idx} mean:", np.mean(real_energies[-5:]))

decoder_train.save_weights(model_name)
np.save(
    f"{model_name}.npy",
    np.array(list(zip(real_energies, imag_energies)), dtype=np.complex128),
)

energy = []
REPS = 10
for rep in range(REPS):
    initial_states = np.zeros((50, 1))

    samples, _, _ = decoder_train.autoregressive_sampling(initial_states)

    er, ei = tfim_e_loc(samples, decoder_train)

    energy.append(er)

sns.set_style("darkgrid")
plt.plot(np.array(real_energies))
plt.title(f"Final Mean Energy: {np.mean(energy)}")
plt.savefig(f"{model_name}.png")
