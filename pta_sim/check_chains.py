#!/usr/bin/env python
# coding: utf-8

# Code for checking chain convergence
import numpy as np
import la_forge.core as co
import la_forge.diagnostics as dg
import la_forge
import argparse
import enterprise_extensions.model_utils as eemu

parser = argparse.ArgumentParser()
parser.add_argument('--dir', dest='dir', action='store',
                    type=str, default='.',
                    help='Directory of chain to check.')
parser.add_argument('--hm', dest='hm', action='store_true',
                    default=False,
                    help='Whether or not run is a hypermodel run.')
parser.add_argument('--save', dest='save', action='store_true',
                    default=False,
                    help='Whether or not to save the chain as a la_forge.core')
args = parser.parse_args()

if args.hm:
    hmc = co.HyperModelCore(chaindir=args.dir)
    hmcs = {}
    for c in ['c0', 'c1', 'c2', 'c3', 'c4']:
        try:
            hmcs[c] = hmc.model_core(['c0', 'c1', 'c2', 'c3', 'c4'].index(c))
        except:
            pass
    print("bf: ", eemu.odds_ratio(hmc.get_param('nmodel'), uncertainty=False))
    # check for convergence in nmodel
    rhat, idx = dg.grubin(hmc)
    print("nmod convergence : ", rhat[-1])
    # check for convergence in all subcores
    for c in list(hmcs.keys()):
        rhat, idx = dg.grubin(hmcs[c])
        print(c)
        print(idx)
    # save HyperModelCore
    if args.save:
        save_as = args.dir.split('/')[-1]
        hmc.save(args.dir+save_as)
elif args.hm is False:
    c0 = co.Core(chaindir=args.dir)
    if args.save:
        save_as = args.dir.split('/')[-1]
        c0.save(args.dir+save_as)
    # check for convergence
    rhat, idx = dg.grubin(c0)
    print(idx)

