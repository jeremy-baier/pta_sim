#!/usr/bin/env python
# coding: utf-8

# Noise model selection on NANOGrav pulsars

import json, pickle, copy
import logging
from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions.hypermodel import HyperModel

import pta_sim.parse_sim as parse_sim
args = parse_sim.arguments()
logging.basicConfig(level=logging.WARNING)

filepath = '/home/jeffrey.hazboun/nanograv/12p5yr_work/noise_model/'
filepath += 'noise_model_selection/no_dmx_pickles/'
filepath += '{0}_ng12p5yr_v3_nodmx_ePSR.pkl'.format(args.psr)
with open(args.pickle, 'rb') as fin:
    psr = pickle.load(fin)

with open(args.model_kwargs_path, 'r') as fin:
    model_kwargs = json.load(fin)

# Add to exponential dips for J1713+0747
                #Model, kernel, DM1, DM2, Scatt1, Scatt2
model_labels = [['A', 'periodic', True, True, True, True],
                ['B', 'periodic', True, True, True, False],
                ['C', 'periodic', True, True, False, True],
                ['D', 'periodic', True, False, True, True],
                ['E', 'periodic', False, True, True, True],
                ]
ptas = {}
all_kwargs = {}
for ii, ent in enumerate(model_labels):
    num_dips = 0
    dm_expdip_tmin = []
    dm_expdip_tmax = []
    dm_expdip_idx = []

    if ent[2]:
        num_dips +=1
        dm_expdip_tmin.append(54700)
        dm_expdip_tmax.append(54850)
        dm_expdip_idx.append(2)
    if ent[3]:
        num_dips +=1
        dm_expdip_tmin.append(57450)
        dm_expdip_tmax.append(57450)
        dm_expdip_idx.append(2)
    if ent[4]:
        num_dips +=1
        dm_expdip_tmin.append(54700)
        dm_expdip_tmax.append(54850)
        dm_expdip_idx.append(4)
    if ent[5]:
        num_dips +=1
        dm_expdip_tmin.append(57450)
        dm_expdip_tmax.append(57450)
        dm_expdip_idx.append(4)

    dip_kwargs = {'dm_expdip':True,
                  'dmexp_sign': 'negative',
                  'num_dmdips':num_dips,
                  'dm_expdip_idx':dm_expdip_idx,
                  'dm_expdip_tmin':dm_expdip_tmin,
                  'dm_expdip_tmax':dm_expdip_tmax,
                  'dmdip_seqname':'ism'}

    kwargs = copy.deepcopy(model_kwargs['5'])
    kwargs.update(dip_kwargs)
    ptas[ii] = model_singlepsr_noise(psr, **kwargs)
    all_kwargs[ii] = kwargs


model_labels = [['A', 'periodic', True, True, False, False],
                ['B', 'periodic', False, False, True, True],
                ['C', 'periodic', True, False, False, True],
                ['D', 'periodic', False, True, True, False],
                ['E', 'periodic', True, False, True, False],
                ['F', 'periodic', False, True, False, True]]

super_model = HyperModel(ptas)

sampler = super_model.setup_sampler(resume=True, outdir=args.outdir,
                                    empirical_distr=args.emp_distr)

model_params = {}
for ky, pta in ptas.items():
    model_params.update({str(ky): pta.param_names})

with open(args.outdir + '/model_params.json', 'w') as fout:
    json.dump(model_params, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

with open(args.outdir + '/model_kwargs.json', 'w') as fout:
    json.dump(model_kwargs, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

with open(args.outdir + '/model_labels.json', 'w') as fout:
    json.dump(model_labels, fout, sort_keys=True,
              indent=4, separators=(',', ': '))

# sampler for N steps
N = args.niter
x0 = super_model.initial_sample()

sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, burn=300000)
