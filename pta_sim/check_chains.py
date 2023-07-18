#!/usr/bin/env python
# coding: utf-8

# Code for checking chain convergence
import numpy as np
import la_forge.core as co
import la_forge
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', dest='dir', action='store',
                    type=str, default='.',
                    help='Directory of chain to check.')
parser.add_argument('--hm', dest='hm', action='store_true',
                    default=False,
                    help='Whether or not run is a hypermodel run.')
args = parser.parse_args()

if args.hm:
    hmc = co.HyperModelCore(chaindir=args.dir)
    hmcs = {}
    core_array = ['c0', 'c1', 'c2', 'c3', 'c4'] 
    for c in core_array:
        try:
            hmcs[c] = hmc.model_core(core_array.index(c))
        except:
            pass

elif args.hm is False:
    c0 = co.Core(chaindir=args.dir)


def grubin(core, M=2, threshold=1.01):
    """
    Gelman-Rubin split R hat statistic to verify convergence.
    See section 3.1 of https://arxiv.org/pdf/1903.08008.pdf.
    Values > 1.1 => recommend continuing sampling due to poor convergence.
    More recently, values > 1.01 => recommend continued sampling due to poor convergence.
    Input:
        core (Core): consists of entire chain file
        pars (list): list of parameters for each column
        M (integer): number of times to split the chain
        threshold (float): Rhat value to tell when chains are good
    Output:
        Rhat (ndarray): array of values for each index
        idx (ndarray): array of indices that are not sampled enough (Rhat > threshold)
    """
    if isinstance(core, list) and len(core) == 2:  # allow comparison of two chains
        data = np.concatenate([core[0].chain, core[1].chain])
    else:
        data = core.chain
    burn = 0
    try:
        data_split = np.split(data[burn:,:-2], M)  # cut off last two columns
    except:
        # this section is to make everything divide evenly into M arrays
        P = int(np.floor((len(data[:, 0]) - burn) / M))  # nearest integer to division
        X = len(data[:, 0]) - burn - M * P  # number of additional burn in points
        burn += X  # burn in to the nearest divisor
        burn = int(burn)

        data_split = np.split(data[burn:,:-2], M)  # cut off last two columns

    N = len(data[burn:, 0])
    data = np.array(data_split)

    # print(data_split.shape)

    theta_bar_dotm = np.mean(data, axis=1)  # mean of each subchain
    theta_bar_dotdot = np.mean(theta_bar_dotm, axis=0)  # mean of between chains
    B = N / (M - 1) * np.sum((theta_bar_dotm - theta_bar_dotdot)**2, axis=0)  # between chains

    # do some clever broadcasting:
    sm_sq = 1 / (N - 1) * np.sum((data - theta_bar_dotm[:, None, :])**2, axis=1)
    W = 1 / M * np.sum(sm_sq, axis=0)  # within chains
    
    var_post = (N - 1) / N * W + 1 / N * B
    Rhat = np.sqrt(var_post / W)

    idx = np.where(Rhat > threshold)[0]  # where Rhat > threshold
    return Rhat, idx

if args.hm:
    for c in list(hmcs.keys()):
        rhat, idx = grubin(hmcs[c])
        print(c)
        print(idx)
elif args.hm is False:
    rhat, idx = grubin(c0)
    print(idx)

