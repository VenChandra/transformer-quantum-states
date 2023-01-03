"""Module for computing observables of decoder ground state
Includes x,z expectation value; xx,zz two-point function,
and nth Renyi entropy.

We estimate correlation functions via importance sampling,
just as we did for energy. See {insert our paper} for
explicit formulas.
"""
import sys
import json
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from TransformerRelativeWF import TransformerDecoder, labels_to_spins

np_config.enable_numpy_behavior()

# IMPORTANT: set flag to False for TFIM if using stoquastic
include_phases = True


def x_correlator(
    idx1: int, idx2: int, sample_size: int, decoder: TransformerDecoder
) -> np.ndarray:
    """
    Args:
        idx1, idx2: int, positions to evaluate two-point fn
        sample_size: int, batch size for importance sampling
        decoder: TransformerDecoder, the transformer wavefunction
    Returns:
        np.ndarray, xx two-point function
    """
    if idx1 == idx2:
        return 1

    # to avoid overloading the memory

    max_batch_size = 15_000
    count = sample_size

    correlator = 0

    while count > 0:
        batch_size = min(max_batch_size, count)
        count -= batch_size

        initial_states = np.zeros((batch_size, 1))

        samples, log_probs, phases = decoder.autoregressive_sampling(initial_states)

        samples = labels_to_spins(decoder.dictionary_size, samples)

        # Obtain the flipped states

        flip_array = np.ones_like(samples)
        flip_array[:, idx1] = -1
        flip_array[:, idx2] = -1

        flipped_samples = ((2 * samples - 1) * flip_array + 1) / 2

        flipped_log_probs, flipped_phases = decoder.evaluate_state(flipped_samples,)

        ratio_mags = tf.math.exp((1 / 2) * (flipped_log_probs - log_probs))

        if include_phases:
            ratio_real_part = ratio_mags * tf.math.cos(-phases + flipped_phases)
        else:
            ratio_real_part = ratio_mags

        # we only care about the real part

        correlator += np.sum(ratio_real_part.numpy())

    return correlator / sample_size


def x_expectation(
    idx: int, sample_size: int, decoder: TransformerDecoder
) -> np.ndarray:
    """
    Args:
        idx: int, position to evaluate expectation
        sample_size: int, number of samples
        decoder: TransformerDecoder, the transformer wavefunction
    Returns:
        np.ndarray, <sigma_x> at idx via importance sampling
    """

    # to avoid overloading the memory

    max_batch_size = 15_000
    count = sample_size

    expectation = 0

    while count > 0:
        batch_size = min(max_batch_size, count)
        count -= batch_size

        initial_states = np.zeros((batch_size, 1))

        samples, log_probs, phases = decoder.autoregressive_sampling(initial_states)

        # Obtain the flipped states

        flip_array = np.ones_like(samples)
        flip_array[:, idx] = -1

        flipped_samples = ((2 * samples - 1) * flip_array + 1) / 2

        flipped_log_probs, flipped_phases = decoder.evaluate_state(flipped_samples,)

        ratio_mags = tf.math.exp((1 / 2) * (flipped_log_probs - log_probs))

        if include_phases:
            ratio_real_part = ratio_mags * tf.math.cos(-phases + flipped_phases)
        else:
            ratio_real_part = ratio_mags

        # we only care about the real part

        expectation += np.sum(ratio_real_part.numpy())

    return expectation / sample_size


def z_correlator(sample_size: int, decoder: TransformerDecoder) -> np.ndarray:
    """
    Args:
        sample_size: int, batch size for importance sampling
        decoder: TransformerDecoder, the transformer wavefunction
    Returns:
        np.ndarray, matrix of <sigma_z sigma_z> between all pairs of positions
    """

    # to avoid overloading the memory

    max_batch_size = 15_000
    count = sample_size

    length = int((decoder.sequence_length - 1) * np.log2(decoder.dictionary_size))

    correlations = np.zeros((length, length))

    while count > 0:
        batch_size = min(max_batch_size, count)
        count -= batch_size

        initial_states = np.zeros((batch_size, 1))

        samples, _, _ = decoder.autoregressive_sampling(initial_states)

        samples = labels_to_spins(decoder.dictionary_size, samples)

        # compute correlator

        inner = np.einsum("bi,bj->ij", 2 * samples - 1, 2 * samples - 1)

        correlations += inner

    return correlations / sample_size


def z_expectation(
    idx: int, sample_size: int, decoder: TransformerDecoder
) -> np.ndarray:
    """
    Args:
        idx: int, position to evaluate expectation
        sample_size: int, the number of samples
        decoder: TransformerDecoder, the transformer wavefunction
    Returns:
        np.ndarray, <sigma_z> at idx
    """

    # to avoid overloading the memory

    max_batch_size = 15_000
    count = sample_size

    correlator = 0

    while count > 0:
        batch_size = min(max_batch_size, count)
        count -= batch_size

        initial_states = np.zeros((batch_size, 1))

        samples, _, _ = decoder.autoregressive_sampling(initial_states)

        samples = labels_to_spins(decoder.dictionary_size, samples)

        # compute correlator

        spins = 1 - 2 * samples[:, idx]

        correlator += np.sum(spins)

    return correlator / sample_size


def compute_nth_renyi(
    decoder: TransformerDecoder,
    n: int,
    batch_size: int,
    region: Tuple[int, int],
    reps: int = 1,
) -> float:
    """Compute an importance sampling estimate of the nth renyi entropy
    for reduced density matrix of region, using the replica trick.

    See Appendix E of https://arxiv.org/pdf/2002.02973.pdf.

    Args:
        decoder: TransformerDecoder, transformer wavefunction
        n: int, degree of Renyi entropy
        batch_size: int, number of samples
        region: Tuple[int,int], subregion for entropy
        reps: int, number of repetitions of importance sampling
    Returns:
        float, nth Renyi entropy for given region
        """
    assert isinstance(n) == int, f"expected n to be an int, recieved {type(n)}"
    assert n >= 2, f"expected n>=2, recieved n={n}"
    assert region[0] < region[1]

    total_batch_size = batch_size * n
    seed_state = np.zeros((total_batch_size, 1))

    expectations = []

    for _ in range(reps):

        # sample autoregressively

        (
            states,
            log_probabilities,
            autoregressive_phases,
        ) = decoder.autoregressive_sampling(seed_state)

        # reshape these to have the correct shape
        # states will have shape (batch_size, n, sequence_length-1)

        states = np.reshape(
            states, newshape=(batch_size, n, decoder.sequence_length - 1)
        )

        # log_probabilities and autoregressive_phases will have shape (batch_size, n)

        log_probabilities = np.reshape(log_probabilities, newshape=(batch_size, n))
        autoregressive_phases = np.reshape(
            autoregressive_phases, newshape=(batch_size, n)
        )

        permuted_states = np.roll(states, axis=1, shift=-1,)

        # Now reassign those elements which are outside of the region

        permuted_states[:, :, : region[0]] = states[:, :, : region[0]]
        permuted_states[:, :, region[1] :] = states[:, :, region[1] :]

        permuted_log_probabilities, permuted_total_phases = decoder.evaluate_state(
            np.reshape(
                permuted_states, newshape=(batch_size * n, decoder.sequence_length - 1)
            )
        )

        permuted_log_probabilities = np.reshape(
            permuted_log_probabilities, newshape=(batch_size, n)
        )
        permuted_total_phases = np.reshape(
            permuted_total_phases, newshape=(batch_size, n)
        )

        # now compute the current contribution to the renyi estimate

        all_ratios = np.exp(
            np.sum(permuted_log_probabilities - log_probabilities, -1) / 2
        )
        all_phases = np.exp(
            1j * np.sum(-permuted_total_phases + autoregressive_phases, -1)
        )

        batch_expectation = np.mean(all_ratios * all_phases)

        expectations.append(batch_expectation)

    return np.mean(expectations)


# load names of weights across a training ensemble

for i, param in enumerate(sys.argv):
    if param in ("--name", "-n"):
        model_name = sys.argv[i + 1]
    if param in ("--ensemble", "-e"):
        ensemble_size = int(sys.argv[i + 1])

# example computation of xx and zz,
# for ensemble of odd integer seeds
# modify as needed

if __name__ == "main":

    # seeds swept over odd integers,
    # change as needed to desired
    # ensemble of seeds

    names = [f"{model_name}_seed_{i}" for i in range(1, 2 * ensemble_size, 2)]

    all_zz_correlations = []
    all_xx_correlations = []

    # load weights for each seed and compute

    for name in names:

        with open(f"params_{name}.json", "r") as param_file:
            model_params = json.loads(param_file.read())

        decoder_model = TransformerDecoder(**model_params)
        decoder_model.load_weights(name)

        SAMPLE_SIZE = 150_000
        SYSTEM_SIZE = int(
            (decoder_model.sequence_length - 1) * np.log2(decoder_model.dictionary_size)
        )

        xx_correlators = []

        # compute <\sigma_x(1) \sigma_x(idx)>

        for i in range(2, SYSTEM_SIZE):
            xx_correlators.append(x_correlator(1, i, SAMPLE_SIZE, decoder_model))

        np.save(f"{name}_xx_correlators", xx_correlators)

        zz_correlators = z_correlator(SAMPLE_SIZE, decoder_model)

        np.save(f"{name}_zz_correlators", zz_correlators)

        all_zz_correlations.append(zz_correlators)
        all_xx_correlations.append(xx_correlators)

    # average over ensemble

    all_zz_correlations = np.array(all_zz_correlations).reshape(
        (len(names), SYSTEM_SIZE, -1)
    )
    all_xx_correlations = np.array(all_xx_correlations).reshape(
        (len(names), SYSTEM_SIZE)
    )
    averaged_zz_correlations = np.mean(all_zz_correlations[:, 25, :], axis=0)
    averaged_xx_correlations = np.mean(all_xx_correlations, axis=0)

    np.save(f"{model_name}_averaged_zz", averaged_zz_correlations)
    np.save(f"{model_name}_averaged_xx", averaged_xx_correlations)
