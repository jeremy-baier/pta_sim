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

elif args.hm is False:
    c0 = co.Core(chaindir=args.dir)

rhat, idx = la_forge.diagnostics.grubin(c0)
print(rhat)





