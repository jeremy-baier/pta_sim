#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys, os, glob, json, pickle, copy
import cloudpickle
import logging

from enterprise_extensions import models, model_utils, hypermodel, sampler
from enterprise.signals.signal_base import PTA
from enterprise.signals import gp_signals, signal_base, deterministic_signals, parameter, selections, white_signals, utils
from enterprise.signals import gp_bases as gpb
from enterprise.signals import gp_priors as gpp
from enterprise import constants as const

from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions import blocks
from enterprise_extensions import model_orfs
from enterprise_extensions import gp_kernels as gpk
from enterprise_extensions import chromatic as chrom
from enterprise_extensions.hypermodel import HyperModel
import la_forge.core as co

import pta_sim.parse_sim as parse_sim
from pta_sim.bayes import chain_length_bool, save_core, get_freqs, filter_psr_path
args = parse_sim.arguments()

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
os.makedirs(args.outdir,exist_ok = True)

with open(args.noisepath, 'r') as fin:
    noise =json.load(fin)

if os.path.exists(args.pta_pkl):
    print("Loading in PTA from pickle file...")
    with open(args.pta_pkl, "rb") as f:
        ptas = cloudpickle.load(f)
else:
    with open('{0}'.format(args.pickle), "rb") as f:
        pkl_psrs = pickle.load(f)

    with open('{0}'.format(args.pickle_nodmx), "rb") as f:
        nodmx_psrs = pickle.load(f)

    adv_noise_psr_list = ['B1855+09', #32
                          'B1937+21', #42
                          'J0030+0451',# #1.4 **
                          'J0613-0200',# -25 *
                          'J0645+5158',# 28 *
                          'J1012+5307',#38
                          'J1024-0719', #-16 **
                          'J1455-3330', #-16 **
                          'J1600-3053', #-10 **
                          'J1614-2230', #-1 **
                          'J1640+2224', #44
                          'J1713+0747', #30 *
                          'J1738+0333', #-26 *
                          'J1741+1351', #37
                          'J1744-1134', #12 **
                          'J1909-3744', #15 **
                          'J1910+1256', #35
                          'J2010-1323', #6 **
                          'J2043+1711',#40
                          'J2317+1439'] #17 *
    #toggle between the full adv noise list and the alt pol psr subsets
    if args.alt_pol_psrs_only:
        adv_noise_psr_list = ['J0030+0451',# #1.4 **
                              'J0613-0200',]# -25 * 
    if args.J0613_only:
        adv_noise_psr_list = ['J0613-0200']
        if args.alt_pol_psrs_only or args.J0030_only:
            ValueError("Conflicting ANM pulsar lists")
    if args.J0030_only:
        adv_noise_psr_list = ['J0030+0451'] 
        if args.alt_pol_psrs_only or args.J0613_only:
            ValueError("Conflicting ANM pulsar lists")
    print("Setting up advanced noise modeling for pulsars...")
    print(adv_noise_psr_list)

    #set up some infrastructure for the hypermodel 
    model_labels = []
    ptas = {}

    def dm_exponential_dip(tmin, tmax, idx=2, sign='negative', name='dmexp', vary=True):
        """
        Returns chromatic exponential dip (i.e. TOA advance):

        :param tmin, tmax:
            search window for exponential dip time.
        :param idx:
            index of radio frequency dependence (i.e. DM is 2). If this is set
            to 'vary' then the index will vary from 1 - 6
        :param sign:
            set sign of dip: 'positive', 'negative', or 'vary'
        :param name: Name of signal
        :param vary: Whether to vary the parameters or use constant values.

        :return dmexp:
            chromatic exponential dip waveform.
        """
        if vary:
            t0_dmexp = parameter.Uniform(tmin, tmax)
            log10_Amp_dmexp = parameter.Uniform(-10, -2)
            log10_tau_dmexp = parameter.Uniform(0, 2.5)
        else:
            t0_dmexp = parameter.Constant()
            log10_Amp_dmexp = parameter.Constant()
            log10_tau_dmexp = parameter.Constant()

        if sign == 'vary' and vary:
            sign_param = parameter.Uniform(-1.0, 1.0)
        elif sign == 'vary' and not vary:
            sign_param = parameter.Constant()
        elif sign == 'positive':
            sign_param = 1.0
        else:
            sign_param = -1.0
        wf = chrom.chrom_exp_decay(log10_Amp=log10_Amp_dmexp,
                             t0=t0_dmexp, log10_tau=log10_tau_dmexp,
                             sign_param=sign_param, idx=idx)
        dmexp = deterministic_signals.Deterministic(wf, name=name)

        return dmexp

    # timing model
    tm = gp_signals.MarginalizingTimingModel()
    #s = gp_signals.MarginalizingTimingModel()
    
    # intrinsic red noise
    # 30 frequencies are appropriate here. see 15yr paper section 3
    s = blocks.red_noise_block(prior='log-uniform', Tspan=args.tspan, components=30)
    #rn  = gp_signals.FourierBasisGP(fs,components=30,Tspan=args.tspan, name='excess_noise')

    m = s 
    #plaw + rn

    # adding white-noise, separating out Adv Noise Psrs, and acting on psr objects
    final_psrs = []
    psr_models = []
    ### Add a stand alone SW deter model
    bins = np.arange(53215, 59200, 180)
    bins *= 24*3600 #Convert to secs
    # n_earth = chrom.solar_wind.ACE_SWEPAM_Parameter(size=bins.size-1)('n_earth')
    if args.sw_fit_path is None:
        n_earth = parameter.Uniform(0,30,size=bins.size-1)('n_earth')
        np_earth = parameter.Uniform(-4, -2)('np_4p39')
    else:
        n_earth = parameter.Constant()('n_earth')
        np_earth = parameter.Constant()('np_4p39')
        with open(args.sw_fit_path,'r') as fin:
            sw_vals = json.load(fin)
        noise.update(sw_vals)

    deter_sw = chrom.solar_wind.solar_wind(n_earth=n_earth, n_earth_bins=bins)
    mean_sw = deterministic_signals.Deterministic(deter_sw, name='sw_r2')

    if args.sw_r4p4:
        sw_power = parameter.Constant(4.39)('sw_power_4p39')
        deter_sw_p = chrom.solar_wind.solar_wind_r_to_p(n_earth=np_earth,
                                                        power=sw_power,
                                                        log10_ne=True)
        mean_sw += deterministic_signals.Deterministic(deter_sw_p,
                                                       name='sw_4p39')

    if args.orf2 == "None":
        args.orf2=None

    cs_alt_pol = blocks.common_red_noise_block(psd=args.psd,
                                        prior='log-uniform',
                                        Tspan=args.tspan,
                                        #orf=model_orfs.st_orf(),
                                        orf=args.orf2,
                                        components=args.n_gwbfreqs,
                                        gamma_val=args.gamma_gw,
                                        name='gw')
    cs = blocks.common_red_noise_block(psd=args.psd,
                                        prior='log-uniform',
                                        Tspan=args.tspan,
                                        #orf=model_orfs.hd_orf(),
                                        orf = args.orf,
                                        components=args.n_gwbfreqs,
                                        gamma_val=args.gamma_gw,
                                        name='gw')
    ##### below loops over pulsars if statements separate out dmx and no_dmx
    for psr,psr_nodmx in zip(pkl_psrs,nodmx_psrs):
        # Filter out other Adv Noise Pulsars
        if psr.name in adv_noise_psr_list:
            new_psr = psr_nodmx
            ### Get kwargs dictionary
            kwarg_path = args.model_kwargs_path
            kwarg_path += f'{psr.name}_model_kwargs.json'
            with open(kwarg_path, 'r') as fin:
                kwargs = json.load(fin)

            ### Turn SW model off. Add in stand alone SW model and common process. Return model.
            kwargs.update({'dm_sw_deter':False,
                            'white_vary':args.vary_wn,
                            'extra_sigs':m + mean_sw,
                            'psr_model':True,
                            'chrom_df':None,
                            'dm_df':None,
                            'red_var': False,
                            'tm_marg':True,
                            'vary_dm':False,
                            'tm_svd':False,
                            'vary_chrom':False})
            ### Load the appropriate single_pulsar_model
            psr_models.append(model_singlepsr_noise(new_psr, **kwargs))#(new_psr))
            final_psrs.append(new_psr)
        # Treat all other DMX pulsars in the standard way
        elif not args.adv_noise_psrs_only:
            s2 = s + tm + blocks.white_noise_block(vary=False,
                                                   tnequad=False,
                                                   inc_ecorr=True,
                                                   select='backend')
            psr_models.append(s2)#(psr))
            final_psrs.append(psr)

        print(f'\r{psr.name} Complete.',end='',flush=True)

    hd_models = [(m + cs)(psr) for psr,m in  zip(final_psrs,psr_models)]
    alt_pol_models = [(m + cs_alt_pol)(psr) for psr,m in  zip(final_psrs,psr_models)]

    pta_hd = signal_base.PTA(hd_models)
    pta_hd.set_default_params(noise)
    pta_alt_pol = signal_base.PTA(alt_pol_models)
    pta_alt_pol.set_default_params(noise)

    ptas = {0:pta_hd,
             1:pta_alt_pol}
    model_labels.append(['0', args.orf])
    model_labels.append(['1', args.orf2])
    # make a dictionary in the file for model labels and model params
    model_params = {}
    for ky, pta in ptas.items():
        model_params.update({str(ky) : pta.param_names})
    with open(args.outdir+'/model_params.json' , 'w') as fout:
        json.dump(model_params, fout, sort_keys=True,
              indent=4, separators=(',', ': '))
    with open(args.outdir+'/model_labels.json' , 'w') as fout:
        json.dump(model_labels, fout, sort_keys=True,
              indent=4, separators=(',', ': '))
        


    if args.mk_ptapkl:
        print("Saving pickled pta to file ... ")
        with open(args.pta_pkl,'wb') as fout:
            cloudpickle.dump(ptas,fout)

# here we put together the hyper_model in its full glory
super_model = HyperModel(ptas)
groups = super_model.get_parameter_groups()
#i removed pta_curn as arg from setup_sampler and groups=groups
print("Setting up hypermodel ...")
Sampler = super_model.setup_sampler(outdir=args.outdir, 
                                    resume=True,
                                    empirical_distr = args.emp_distr, 
                                    human = "jeremy",
                                    groups=groups)
    

def draw_from_sw_prior(self, x, iter, beta):

    q = x.copy()
    lqxy = 0

    signal_name = 'sw_r2'

    # draw parameter from signal model
    param = np.random.choice(self.snames[signal_name])
    if param.size:
        idx2 = np.random.randint(0, param.size)
        q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

    # scalar parameter
    else:
        q[self.pmap[str(param)]] = param.sample()

    # forward-backward jump probability
    lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
            param.get_logpdf(q[self.pmap[str(param)]]))

    return q, float(lqxy)

def draw_from_sw4p39_prior(self, x, iter, beta):

    q = x.copy()
    lqxy = 0

    signal_name = 'sw_4p39'

    # draw parameter from signal model
    param = np.random.choice(self.snames[signal_name])
    if param.size:
        idx2 = np.random.randint(0, param.size)
        q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

    # scalar parameter
    else:
        q[self.pmap[str(param)]] = param.sample()

    # forward-backward jump probability
    lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
            param.get_logpdf(q[self.pmap[str(param)]]))

    return q, float(lqxy)

def draw_from_gw_gamma_prior(self, x, iter, beta):

    q = x.copy()
    lqxy = 0

    # draw parameter from signal model
    signal_name = [par for par in self.pnames
                   if ('gw' in par and 'gamma' in par)][0]
    idx = list(self.pnames).index(signal_name)
    param = self.params[idx]

    q[self.pmap[str(param)]] = np.random.uniform(param.prior._defaults['pmin'], param.prior._defaults['pmax'])

    # forward-backward jump probability
    lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
            param.get_logpdf(q[self.pmap[str(param)]]))

    return q, float(lqxy)


if args.psd =='powerlaw' and args.gamma_gw is None:
    sampler.JumpProposal.draw_from_gw_gamma_prior = draw_from_gw_gamma_prior
    Sampler.addProposalToCycle(Sampler.jp.draw_from_gw_gamma_prior, 25)


try:
    achrom_freqs = get_freqs(ptas, signal_id='gw')
    np.savetxt(args.outdir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')
except:
    pass

if args.ladderpath is not None:
    ladder = np.loadtxt(args.ladderpath)
else:
    ladder = None

print('Signal Names', Sampler.jp.snames)

print("Drawing initial sample from the prior...")
x0 = super_model.initial_sample()
print("Beginning to sample...")
Sampler.sample(x0, args.niter, ladder=ladder, SCAMweight=200, AMweight=100,
               DEweight=200, burn=3000, writeHotChains=args.writeHotChains,
               hotChain=args.hot_chain, Tskip=100, Tmax=args.tempmax)
