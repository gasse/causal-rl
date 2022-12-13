import os
import sys
import pathlib
import json
import argparse
import numpy as np


if __name__ == '__main__':

    # read experiment config
    with open("experiments/toy1/config.json", "r") as json_data_file:
        cfg = json.load(json_data_file)

    # read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--seed',
        type=int,
        help = 'Random generator seed.',
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        type=int,
        help='CUDA GPU id (-1 for CPU).',
        default=-1,
    )
    parser.add_argument(
        'privileged_policy',
        type=str,
        choices=cfg['privileged_policies'].keys(),
    )
    args = parser.parse_args()

    # process command-line arguments
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu}"
        device = f"cuda:{args.gpu}"

    seed = args.seed
    privileged_policy = args.privileged_policy

    print(f"device: {device}")
    print(f"seed: {seed}")
    print(f"privileged_policy : {privileged_policy}")


    import torch

    # Ugly hack
    sys.path.insert(0, os.path.abspath(f"."))

    from environments import TabularPOMDP
    from policies import UniformPolicy, PrivilegedPolicy
    from models import TabularAugmentedModel

    from utils import construct_dataset
    from learning import fit_model


    ## SET UP THE ENVIRONMENT ##

    p_s = torch.tensor(cfg['p_s'])
    p_r_s = torch.tensor(cfg['p_r_s'])
    p_o_s = torch.tensor(cfg['p_o_s'])
    p_s_sa = torch.tensor(cfg['p_s_sa'])

    p_a_s = torch.tensor(cfg['privileged_policies'][privileged_policy])

    o_nvals=p_o_s.shape[1]
    a_nvals=p_s_sa.shape[1]
    r_nvals=p_r_s.shape[1]
    s_nvals = cfg["latent_space_size"]

    episode_length = cfg["episode_length"]

    # POMDP dynamics
    env = TabularPOMDP(p_s=p_s,
                   p_or_s=p_r_s.unsqueeze(-2) * p_o_s.unsqueeze(-1),
                   p_s_sa=p_s_sa,
                   episode_length=episode_length)

    # Policy in the observational regime (privileged)
    obs_policy = PrivilegedPolicy(p_a_s)

    # Policy in the interventional regime
    int_policy = UniformPolicy(a_nvals)


    ## SET UP THE SEEDS ##

    rng = np.random.RandomState(seed)
    seed_data_obs = rng.randint(0, 2**10)
    seed_data_int = rng.randint(0, 2**10)
    seed_training = rng.randint(0, 2**10)


    ## GENERATE THE DATASETS ##

    nsamples_obs_subsets = cfg['nsamples_obs']
    nsamples_int_subsets = cfg['nsamples_int']

    # we perform experiments on subsets of the same
    # dataset, so that each sequentially growing experiment
    # reuses the same samples, complemented with new ones
    nsamples_obs_total = np.max(nsamples_obs_subsets)
    nsamples_int_total = np.max(nsamples_int_subsets)

    torch.manual_seed(seed_data_obs)
    data_obs_all = construct_dataset(env=env,
        policy=obs_policy,
        n_samples=nsamples_obs_total,
        privileged=True)

    torch.manual_seed(seed_data_int)
    data_int_all = construct_dataset(env=env,
        policy=int_policy,
        n_samples=nsamples_int_total,
        privileged=False)


    ## LEARN THE TRANSITION MODELS ##

    loss_type = 'nll'
    with_done = False

    n_epochs = 500
    epoch_size = 50
    batch_size = 32
    lr = 1e-2
    patience = 10

    device = torch.device(device)

    training_schemes = cfg["training_schemes"]

    for nsamples_obs in nsamples_obs_subsets:
         for nsamples_int in nsamples_int_subsets:

            data_obs = data_obs_all[:nsamples_obs]
            data_int = data_int_all[:nsamples_int]

            modeldir = pathlib.Path(f"experiments/toy1/trained_models/{privileged_policy}/seed_{seed}/nobs_{nsamples_obs}/nint_{nsamples_int}")
            modeldir.mkdir(parents=True, exist_ok=True)

            print(f"saving results to: {modeldir}")

            for training_scheme in training_schemes:

                print(f"nsamples_obs: {nsamples_obs} nsamples_int: {nsamples_int} training_scheme: {training_scheme}")

                logfile = modeldir / f"{training_scheme}_log.txt"
                paramsfile = modeldir / f"{training_scheme}.pt"

                if training_scheme == 'int':
                    train_data = data_int
                elif training_scheme == 'obs+int':
                    train_data = [(torch.tensor(1), episode) for (_, episode) in data_obs + data_int]
                elif training_scheme == 'augmented_obs+int':
                    train_data = data_obs + data_int
                else:
                    raise NotImplemented

                torch.manual_seed(seed_training)

                m = TabularAugmentedModel(s_nvals=s_nvals, o_nvals=o_nvals, a_nvals=a_nvals, r_nvals=r_nvals)
                m = m.to(device)

                fit_model(m,
                          train_data=train_data,
                          valid_data=train_data,  # we want to overfit
                          loss_type=loss_type,
                          with_done=with_done,
                          n_epochs=n_epochs,
                          epoch_size=epoch_size,
                          batch_size=batch_size,
                          lr=lr,
                          patience=patience,
                          log=True,
                          logfile=logfile)

                torch.save(m.state_dict(), paramsfile)

