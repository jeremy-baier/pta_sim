#### example run from the command line
# python3 /home/zeus/baierj/src/pta_sim/pta_sim/scripts/resampler.py\
# --outdir /home/zeus/baierj/outdir/resampler_results --nsamps 1000 --model_core 0\
# --pta_path /home/zeus/baierj/run_files/dallas_files/CURN_v_ST_65psr/pickled_ptas.pkl --pta_index 1 --save_as test1\
# --chain_dir /home/zeus/baierj/run_files/dallas_files/CURN_v_ST_65psr\
####

import numpy as np
import la_forge.core as co
import cloudpickle as cp
import multiprocess as mp
from statistics import stdev
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--core_path', dest='core_path', action='store',
                    type=str, default=None,
                    help='Path to la_forge core.')
parser.add_argument('--aux_core', dest='aux_core', action='store',
                    type=str, default=None,
                    help='Path to an additional core.')
parser.add_argument('--chain_dir', dest='chain_dir', action='store',
                    type=str, default=None,
                    help='Instead of passing a core, pass the chain directory for the approx distr.')
parser.add_argument('--pta_path', dest='pta_path', action='store',
                    type=str, default=None,
                    help='Path to pickled pta object.')
parser.add_argument('--nsamps', dest='nsamps', action='store',
                    type=int, default=None,
                    help='approximate number of samples to use in resampling.')
parser.add_argument('--model_core', dest='model_core', action='store',
                    type=int, default=None,
                    help='if using a hypermodel, which sub core to use.\
                        use a negative number to concatenate across cores')
parser.add_argument('--pta_index', dest='pta_index', action='store',
                    type=int, default=None,
                    help='if using a hypermodel, which pta_likelihood to use')
parser.add_argument('--proc', dest='proc', action='store',
                    type=int, default=1,
                    help='number of processors to run on')
parser.add_argument('--outdir', dest='outdir', action='store',
                    type=str, default='./',
                    help='directory to save data to')
parser.add_argument('--save_as', dest='save_as', action='store',
                    type=str, default="unnamed_resampled_data",
                    help='file name to save output to')

args = parser.parse_args()

# load in la_forge core of approximate distribution from either a hypermodel core or a single core
assert args.core_path or args.chain_dir is not None
#assert args.core_path or args.chain_dir is None
if args.model_core is None:
    if args.chain_dir is None:
        core = co.Core(corepath=args.core_path)
    elif args.core_path is None:
        core = co.Core(chaindir=args.chain_dir)
    if args.aux_core is not None:
        aux_core = co.Core(corepath=args.aux_core)
elif args.model_core >= 0:
    if args.chain_dir is None:
        hmc = co.HyperModelCore(corepath=args.core_path)
        core = hmc.model_core(args.model_core)
    elif args.core_path is None:
        hmc = co.HyperModelCore(chaindir=args.chain_dir)
        core = hmc.model_core(args.model_core)
    if args.aux_core is not None:
        aux_hmc = co.HyperModelCore(corepath=args.aux_core)
        aux_core = aux_hmc.model_core(args.model_core)
# FIXME: i dont think the below actually makese sense
elif args.model_core <= 0:
    hmc = co.HyperModelCore(corepath=args.core_path)
    core1 = hmc.model_core(0) 
    core2 = hmc.model_core(1)



# load in the pta from pickle file
with open(args.pta_path, "rb") as f:
    pta = cp.load(f)
# if appropriate, select pta_likelihood to use
if args.pta_index is not None:
    pta = pta[args.pta_index]


############### resampling functions ####################

def setup_resampler(pta, core, nsamps):
    ### take core posterior samples and put them in a dictionary ###
    if args.nsamps is not None:
        # thin samples to about the length of nsamps
        trim_idx = int(len(core.get_param('lnlike')) / nsamps)
        # create dictionary of thinned samples
        core_samples_dict = { key : core.get_param(key)[::trim_idx] for key in core.params }
    elif args.nsamps is None:
        # use all posterior samples
        core_samples_dict = { key : core.get_param(key)[:] for key in core.params }

    # take the lnlikelihood of the CURN samples and store them in a vector
    approx_likelihoods = core_samples_dict['lnlike']
    # make thinned dictionary into an (parameters, samples) array
    # can pass this array to .get_likelihood provided that the parameters are in the correct order
    core_samples_array = np.array([core_samples_dict[key] for key in core_samples_dict.keys()])
    print("Posterior thinned to ",len(approx_likelihoods)," samples.")
    try:
        x0 = np.array([core_samples_dict[key][1] for key in core_samples_dict.keys()])
        test_target_likelihoods = pta.get_lnlikelihood(x0)
        print("Sucessfully evaluated test likelihood")
    except:
        ValueError("Failed test likelihood calculation")

    return approx_likelihoods, core_samples_dict, core_samples_array


def resampler(core_samples_array, pta, processes=args.proc):
    if processes == 1:
        target_likelihoods = np.array([])
        for i in range(len(core_samples_array['lnlike'][0])):
            target_likelihoods = np.append(
                target_likelihoods,
                pta.get_lnlikelihood({key:core_samples[key][i] for key in core_samples.keys()})
                #pta.get_lnlikelihood(core_samples_array[:,i])
            )
    elif processes > 1:
        # FIXME : doesn't currently work. not sure if needed? 
        # i think numpy will just parallelize this?
        def likelihood_wrapper(*args):
            return pta.get_lnlikelihood([arg for arg in args])
        # uses the multiprocess module which is based off of multiprocessing but has some bug fix
        # sets up pool with number of processes
        pool = mp.Pool(processes=processes)
        # each column is passed to likelihood_wrapper, which
        target_likelihoods = np.array(pool.starmap(likelihood_wrapper, core_samples_array.T))
    
    return target_likelihoods

def resampler_statistics(target_likelihoods, approx_likelihoods):
    resampler_stats = {}
    likelihoods = {'target_likelihoods': target_likelihoods, 'approx_likelihoods': approx_likelihoods}
    # calculate the ln_likelihood ratios.
    likelihoods['ln_likelihood_ratios'] = target_likelihoods - approx_likelihoods
    # the bayes factor is the average of the "weights"
    resampler_stats['ln_bayes_factor'] = sum(likelihoods['ln_likelihood_ratios']) / len(likelihoods['ln_likelihood_ratios'])
    resampler_stats['bayes_factor'] = np.exp(resampler_stats['ln_bayes_factor'])
    print("Bayes factor:  ", resampler_stats['bayes_factor'])
    resampler_stats['Ns'] = len(likelihoods['ln_likelihood_ratios'])
    resampler_stats['sigma_w'] = stdev(likelihoods['ln_likelihood_ratios'])
    resampler_stats['n_eff'] = resampler_stats['Ns'] / ( 1. + ( resampler_stats['sigma_w'] / resampler_stats['bayes_factor'] ) ** 2)
    resampler_stats['efficiency'] = resampler_stats['n_eff'] / resampler_stats['Ns']
    resampler_stats['sigma_bf'] = resampler_stats['sigma_w'] / ( resampler_stats['efficiency'] * resampler_stats['Ns'] ) ** ( 1. / 2. )
    
    return resampler_stats, likelihoods

def save_stats(resampler_stats, outdir=args.outdir, file_name=args.save_as):
    #json cant handle np.array in diciontary so turn them into lists
    for key in list(resampler_stats.keys()):
        if type(resampler_stats[key]) == np.ndarray:
                resampler_stats[key] = list(resampler_stats[key]) 
    with open(outdir+file_name+'.json', 'w') as fp:
        #json.dump(resampler_stats, fp)
        fp.write(json.dumps(resampler_stats))
    print("Saving results to ", outdir+file_name+'.json')
    #with open(outdir+file_name+'.pkl','wb') as fout:
     #       cp.dump(stats, fout)
    return 0


# run resampler functions
if args.model_core is None or args.model_core >= 0 and args.aux_core is None:
    approx_likelihoods, core_samples_dict, core_samples_array = setup_resampler(pta=pta, core=core, nsamps=args.nsamps)
elif args.model_core < 0:
   approx1, _, csamps1 = setup_resampler(pta=pta, core=core1, nsamps=np.floor(args.nsamps/2))
   approx2, _, csamps2 = setup_resampler(pta=pta, core=core2, nsamps=np.floor(args.nsamps/2))
   approx_likelihoods = np.append(approx1, approx2)
   core_samples_array = np.append(csamps1, csamps2, axis = 1)
elif args.aux_core is not None:
    pass
else:
    print(" you ducked up somewhere ")
print("Resampling... ")
target_likelihoods = resampler(core_samples_array=core_samples_dict, pta=pta, processes=args.proc)

resampling_stats, likelihoods = resampler_statistics(target_likelihoods=target_likelihoods, approx_likelihoods=approx_likelihoods)

save_stats(resampling_stats)
l_filename = args.save_as + '_likelihoods'
save_stats(likelihoods, file_name=l_filename)
print("Done.")




