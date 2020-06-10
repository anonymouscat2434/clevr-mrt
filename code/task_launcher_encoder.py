import argparse
import torch
import sys
import glob
import os
import numpy as np
import pickle
from torch import nn
from torch.utils.data import (DataLoader,
                              Subset)
from torchvision import datasets
from torchvision.transforms import transforms
from PIL import Image
from torchvision.utils import save_image

from models.resnet_encoder import ResnetEncoder
from models.holo_encoder import HoloEncoder

from functools import partial
from subprocess import check_output
import yaml
import inspect
from collections import OrderedDict
from importlib import import_module

from common import load_dataset

#from handlers import (rot_handler,
#                      kpt_handler,
#                      angle_analysis_handler,
#                      image_handler_default)
from tools import (generate_name_from_args,
                   count_params,
                   line2dict)


if __name__ == '__main__':

    def parse_args():
        parser = argparse.ArgumentParser(description="")
        subparsers = parser.add_subparsers(
            help='Either load args from a config or specify them on the command line',
            dest='which')

        parser_load = subparsers.add_parser('load', help='Run from a YAML file')
        parser_load.add_argument('--config', type=str, default=None)

        # load = load from a yaml file
        # run
        parser_run = subparsers.add_parser('run', help='Run from the command line')
        parser_run.add_argument('--class', type=str, default='resnet_encoder',
                                choices=['resnet_encoder', 'holo_encoder'])
        parser_run.add_argument('--dataset', type=str, default='celeba',
                                choices=['clevr',
                                         'clevr_kiwi',
                                         'clevr_kiwi_cc',
                                         'clevr_kiwi_nocc',
                                         'clevr_kiwi_ae'],
                                help="""
                                celeba = CelebA (64px) (set env var DATASET_CELEBA)
                                celeba_hq = CelebA-HQ (128px) (set env DATASET_CELEBAHQ)
                                """)
        parser_run.add_argument('--arch', type=str,
                                default=None,
                                help="The model which will be used as the encoder")
        parser_run.add_argument('--arch_args', type=str,
                                default=None)
        parser_run.add_argument('--arch_checkpoint', type=str,
                                default=None,
                                help="The encoder model's checkpoint file")
        parser_run.add_argument('--use_holovae', action='store_true',
                                help="""If set, it means our encoder will be from
                                HoloVAE. This means that the only flag you should
                                provide is `arch_args`, and this should be the cfg.yaml
                                of that experiment. In other words, this flag overrides
                                the meaning of `arch_args`.""")
        parser_run.add_argument('--disable_rot', action='store_true')
        parser_run.add_argument('--rot_consist', type=float, default=0.)
        parser_run.add_argument('--subset_train', type=int, default=None,
                                help="""If set, artficially decrease the size of the training
                                data. Use this to easily perform data ablation experiments.""")
        parser_run.add_argument('--probe', type=str,
                                default=None)
        parser_run.add_argument('--probe_args', type=str,
                                default=None)
        parser_run.add_argument("--img_size", type=int, default=224)
        parser_run.add_argument("--imagenet_scaling", action='store_true')
        parser_run.add_argument('--batch_size', type=int, default=32)
        parser_run.add_argument('--epochs', type=int, default=200)
        parser_run.add_argument('--n_channels', type=int, default=3,
                                help="""Number of input channels in image. This is
                                "passed to the `get_network` function defined by
                                "`--arch`.""")
        #parser_run.add_argument('--ngf', type=int, default=64,
        #                        help="""# channel multiplier for the autoencoder. This
        #                        "is passed to the `get_network` function defined by
        #                        `--arch`.""")
        #parser_run.add_argument('--ndf', type=int, default=64,
        #                        help="""# channel multiplier for the discriminator. This
        #                        is passed to the `get_network` function defined by
        #                        `--arch`.""")
        parser_run.add_argument('--cls_loss', type=str, default='cce',
                                choices=['cce', 'mse'],
                                help="""The loss function to use for the probe component. The
                                choices are 'cce' (cat cross-entropy) or 'mse' (mean-squared
                                error).""")
        parser_run.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
        parser_run.add_argument('--beta1', type=float, default=0.9, help="beta1 term of ADAM")
        parser_run.add_argument('--beta2', type=float, default=0.999, help="beta2 term of ADAM")
        parser_run.add_argument('--prior_std', type=float, default=1.0, help="Stdev of prior distribution")
        parser_run.add_argument('--weight_decay', type=float, default=0.0,
                                help="""L2 weight decay on params (note: applies to optimisers for both
                                the generator, discriminator, and classifier probe (if set)""")
        parser_run.add_argument('--seed', type=int, default=0)

        #############

        # HACK: these options need to be added to both subparses

        for parser_obj in [parser_run, parser_load]:

            parser_obj.add_argument('--name', type=str, default=None)
            parser_obj.add_argument('--trial_id', type=str, default=None)
            parser_obj.add_argument('--save_path', type=str, default=None)
            parser_obj.add_argument('--val_batch_size', type=int, default=64)
            parser_obj.add_argument('--valid_seed', type=int, default=0)
            parser_obj.add_argument('--save_every', type=int, default=5)
            parser_obj.add_argument('--save_images_every', type=int, default=1)
            parser_obj.add_argument('--resume', type=str, default='auto')
            parser_obj.add_argument('--load_nonstrict', action='store_true')
            parser_obj.add_argument('--no_verbose', action='store_true')
            parser_obj.add_argument('--num_workers', type=int, default=4)
            parser_obj.add_argument('--no_shuffle', action='store_true')
            parser_obj.add_argument('--mode', type=str,
                                    choices=['train',
                                             'eval_valid',
                                             'eval_test',
                                             'dump_imgs'],
                                    default='train',
                                    help="""
                                    """)
            parser_obj.add_argument('--mode_override', type=str, default=None)

        args = parser.parse_args()
        return args

    args = parse_args()
    args = vars(args)

    # When we export to YAML (so that it can be potentially loaded
    # later with --config <yaml_file>, we don't want to export these
    # options here.
    # TODO: consolidate with parser_obj loop
    DO_NOT_EXPORT = set(['trial_id',
                         'name',
                         'save_path',
                         'val_batch_size',
                         'save_every',
                         'save_images_every',
                         'resume',
                         'load_nonstrict',
                         'no_verbose',
                         'num_workers',
                         'no_shuffle',
                         'mode',
                         'mode_override'])

    # If the config subparser was chosen, then `args` must be deserialised
    # from the file, otherwise continue.
    if args['which'] == 'load':

        loaded_dict = yaml.load(open(args['config']))
        print("Loaded dict: ", loaded_dict)

        for key in loaded_dict:
            # We assume '' or 'true' ==> a boolean.
            if type(loaded_dict[key]) == str:
                if loaded_dict[key].lower() == 'true':
                    loaded_dict[key] = True
                elif loaded_dict[key].lower() == 'false':
                    loaded_dict[key] = False
                elif loaded_dict[key].lower() == 'null':
                    loaded_dict[key] = None

        for key in loaded_dict:
            if key not in DO_NOT_EXPORT:
                print("Loading %s = %s" % (key, loaded_dict[key]))
                args[key] = loaded_dict[key]
            else:
                print("WARNING: key %s is specified in yaml but is ignored" % key)


    print("Arguments passed")
    print("  " + yaml.dump(args).replace("\n", "\n  "))

    name = args['trial_id']

    if args['mode'] == 'train':
        torch.manual_seed(args['seed'])
    #else:
    #    torch.manual_seed(0)

    use_cuda = True if torch.cuda.is_available() else False
    cache = True if 'CACHE_DATASET' in os.environ else False

    # If arch is None and not holovae, we assume we want
    # imagenet scaling. Otherwise, the argument
    # --imagenet_scaling will define this.
    if args['arch'] is None and not args['use_holovae']:
        # Assume we're using imagenet pretrained
        print("arch==None and use_holovae==False, so use imagenet scaling...")
        imagenet_scaling = True
    else:
        if not args['imagenet_scaling']:
            imagenet_scaling = False
        else:
            print("imagenet_scaling==True, so use imagenet scaling...")
            imagenet_scaling = True

    if args['mode'] == 'eval_test':
        ds_train, ds_valid = load_dataset('blank',
                                          args['img_size'])
    else:
        ds_train, ds_valid = load_dataset(
            name=args['dataset'],
            img_size=args['img_size'],
            imagenet_scaling=imagenet_scaling
        )


    if args['subset_train'] is not None:
        # The subset is randomly sampled from the
        # training data, and changes depending on
        # the seed.
        indices = np.arange(0, args['subset_train'])
        rs = np.random.RandomState(args['seed'])
        rs.shuffle(indices)
        indices = indices[0:args['subset_train']]
        old_ds_train = ds_train
        ds_train = Subset(old_ds_train, indices=indices)
        # Transfer over vocab file
        ds_train.vocab = old_ds_train.vocab


    if args['mode'] == 'train':
        bs = args['batch_size']
    else:
        bs = args['val_batch_size']
    loader_train = DataLoader(ds_train,
                              batch_size=bs,
                              shuffle=False if args['no_shuffle'] else True,
                              num_workers=args['num_workers'])
    loader_valid = DataLoader(ds_valid,
                              batch_size=bs,
                              shuffle=False,
                              num_workers=args['num_workers'])
    loader_test = None
    #if ds_test is not None:
    #    loader_test = DataLoader(ds_test,
    #                             batch_size=bs,
    #                             shuffle=False,
    #                             num_workers=0,
    #                             drop_last=False)

    if args['save_path'] is None:
        args['save_path'] = os.environ['RESULTS_DIR']

    # <save_path>/<seed>/<name>/_trial=<trial>,...,...,
    if args['name'] is None:
        save_path = "%s/s%i" % (args['save_path'], args['seed'])
    else:
        save_path = "%s/s%i/%s" % (args['save_path'], args['seed'], args['name'])
    print("*** SAVE PATH OF THIS EXPERIMENT: ***")
    print(save_path)
    print("*************************************")

    enc = None
    if args['use_holovae']:
        print("`use_holovae=True`, so `args_arch` should be the cfg.yaml" + \
              " of a HoloVAE experiment.")

        import task_launcher

        default_kwargs = dict(task_launcher.DEFAULT_KWARGS_LOAD)
        default_kwargs['name'] = "ignore_me"
        default_kwargs['trial_id'] = "ignore_me"
        default_kwargs['resume'] = args['arch_checkpoint']
        # mode=load == return the gan class
        # TODO: clean up and get rid of these
        # nuisance params.
        new_args ={
            'which': 'load',
            'config': args['arch_args'],
            'mode': 'load'
        }
        default_kwargs.update(new_args)
        enc = task_launcher.do(default_kwargs)

        # This will already be in cuda,
        # so just put it in eval mode.
        enc._eval()

    else:

        if args['arch'] is None:
            """
            https://github.com/ethanjperez/film/blob/master/scripts/extract_features.py
            Based on this: take resnet101, and take up to
            resblock #4.
            """
            print("`arch` is `None`, so defaulting to Resnet-101 pretrained...")
            from torchvision.models import resnet101
            enc = resnet101(pretrained=True)
            layers = [
                enc.conv1,
                enc.bn1,
                enc.relu,
                enc.maxpool,
            ]
            for i in range(3):
                layer_name = 'layer%d' % (i + 1)
                layers.append(getattr(enc, layer_name))
            enc = torch.nn.Sequential(*layers)
        else:
            print("Loading architecture %s..." % \
                  (args['arch']))
            mod = import_module(args['arch'].replace("/", ".").\
                                replace(".py", ""))
            #arch_kwargs = eval(args['arch_args'])
            #print("  Args passed: %s" % arch_kwargs)
            if args['arch_args'] is None:
                print("`arch_args` is empty, so inferring args...")
                print("  Finding cfg.yaml in arch_checkpoint directory...")
                arch_cfg = "%s/cfg.yaml" % (os.path.dirname(args['arch_checkpoint']))
                arch_kwargs = eval(yaml.load(open(arch_cfg).read())['arch_args'])
                print("  Arch args detected from arch cfg.yaml:")
                print("    " + str(arch_kwargs))
            else:
                arch_kwargs = eval(args['arch_args'])
            #arch_kwargs = {}
            dd = mod.get_network(**arch_kwargs)
            gen = dd['gen']
            if args['arch_checkpoint'] is not None:
                chkpt_dat = torch.load(args['arch_checkpoint'])
                print("  Loading checkpoint %s" % args['arch_checkpoint'])
                gen.load_state_dict(chkpt_dat['g'])
            if not hasattr(gen, 'encode'):
                raise Exception("gen must have `encode` method!!")
            print("Encoder:")
            print("  " + str(gen).replace("\n","\n  "))
            print("  # params: %i" % count_params(gen))
            enc = gen

        # Put on GPU and also eval mode.
        enc.cuda()
        enc.eval()

    probe = None
    if args['probe'] is not None:
        mod = import_module(args['probe'].replace("/", ".").\
                            replace(".py", ""))
        print("Importing probe: %s" % args['probe'])
        print("  This takes the following args:")
        print("  " + str(tuple(inspect.getargspec(mod.get_network).args)))
        probe_args = eval(args['probe_args']) if args['probe_args'] is not None else {}
        #if args['arch_checkpoint'] is not None:
        #    print("  `n_in` of probe must match `enc_dim` of arch " + \
        #          "so performing this replacement...")
        #    probe_args['n_in'] = arch_kwargs['enc_dim']

        probe = mod.get_network(ds_train.vocab, **probe_args)
        print("Probe:")
        print("  " + str(probe).replace("\n","\n  "))
        print("  # params: %i" % count_params(probe))
    else:
        if args['class'] == 'resnet_encoder':
            raise Exception("probe must be specified for `resnet_encoder` class!")

    if args['class'] == 'resnet_encoder':
        class_name = ResnetEncoder
    else:
        class_name = HoloEncoder

    net = class_name(
        enc=enc,
        probe=probe,
        disable_rot=args['disable_rot'],
        rot_consist=args['rot_consist'],
        cls_loss=args['cls_loss'],
        opt_args={'lr': args['lr'],
                  'betas': (args['beta1'], args['beta2']),
                  'weight_decay': args['weight_decay']},
        handlers=[]
    )

    if args['load_nonstrict']:
        net.load_strict = False

    latest_model = None
    if args['resume'] is not None:
        if args['resume'] == 'auto':
            # autoresume
            model_dir = "%s/%s" % (save_path, name)
            # List all the pkl files.
            files = glob.glob("%s/*.pkl" % model_dir)
            # Make them absolute paths.
            files = [os.path.abspath(key) for key in files]
            if len(files) > 0:
                # Get creation time and use that.
                latest_model = max(files, key=os.path.getctime)
                print("Auto-resume mode found latest model: %s" %
                      latest_model)
                net.load(latest_model)
        else:
            print("Loading: %s" % args['resume'])
            net.load(args['resume'])

    if latest_model is not None:
        cpt_name = os.path.basename(latest_model)
    else:
        cpt_name = os.path.basename(args['resume'])

    expt_dir = "%s/%s" % (save_path, name)
    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    print("EXPERIMENT DIR: %s" % expt_dir)

    # Save the config to the experiment folder, but
    # only the ones that are needed.
    copied_dict = dict(args)
    for key in DO_NOT_EXPORT:
        del copied_dict[key]
    cfg_file = "%s/cfg.yaml" % expt_dir
    with open(cfg_file, 'w') as f_cfg:
        f_cfg.write(yaml.dump(copied_dict))
    # Log the git branch in another file.
    git_file = "%s/git.txt" % expt_dir
    with open(git_file, 'a' if os.path.exists(git_file) else 'w') as f_git:
        git_branch = check_output("git rev-parse --symbolic-full-name --abbrev-ref HEAD", shell=True)
        git_branch = git_branch.decode('utf-8').rstrip()
        f_git.write("git_branch: %s\n" % git_branch)

    if args['mode'] == 'train':

        net.fit(itr_train=loader_train,
                itr_valid=loader_valid,
                epochs=args['epochs'],
                model_dir=expt_dir,
                result_dir=expt_dir,
                save_every=args['save_every'],
                verbose=False if args['no_verbose'] else True)

    elif args['mode'] == 'dump_imgs':

        batch = iter(loader_train).next()[0]
        from torchvision.utils import save_image

        save_image(batch*0.5 + 0.5, "batch.png")

    elif args['mode'] in ['eval_valid', 'eval_test']:

        from tqdm import tqdm

        if args['mode'] == 'eval_valid':
            print("Evaluating on valid set...")
            loader = loader_valid
            ds = ds_valid
        else:
            print("Evaluating on test set...")
            ds = load_dataset(
                name=args['dataset'],
                img_size=args['img_size'],
                imagenet_scaling=imagenet_scaling,
                train=False
            )
            loader = DataLoader(ds,
                                batch_size=args['val_batch_size'],
                                shuffle=False,
                                num_workers=args['num_workers'])

        net_epoch = net.last_epoch

        # Ok, for each possible camera view, create that valid set
        # and evaluate on it.
        preds = []
        gt = []
        tfs = []
        cams = []
        nc = []
        ns = []
        nm = []
        n_obj = []

        pbar = tqdm(total=len(loader))
        for batch in loader:
            batch = net.prepare_batch(batch)
            pred = net.predict(*batch).argmax(dim=1)
            y_batch = batch[-2]
            meta_batch = batch[-1]
            cam_xy_batch = batch[3][:,0:2]

            preds.append(pred)
            gt.append(y_batch)
            cams.append(cam_xy_batch)
            #dists.append(meta_batch['d_from_cc'])
            tfs += meta_batch['template_filename']
            nc.append(meta_batch['n_color_unique'])
            ns.append(meta_batch['n_shape_unique'])
            nm.append(meta_batch['n_mat_unique'])
            n_obj.append(meta_batch['n_objects'])

            pbar.update(1)
        pbar.close()

        acc_ind = (torch.cat(preds, dim=0) == torch.cat(gt, dim=0)).float().cpu().numpy()
        cams = torch.cat(cams, dim=0).cpu().numpy()
        nc = torch.cat(nc, dim=0).cpu().numpy()
        ns = torch.cat(ns, dim=0).cpu().numpy()
        nm = torch.cat(nm, dim=0).cpu().numpy()
        n_obj = torch.cat(n_obj, dim=0).cpu().numpy()

        with open("%s/%s.%i.csv" % (expt_dir, args['mode'], net_epoch), "w") as f:
            f.write("correct,cam_x,cam_y,tf,n_color_unique,n_shape_unique,n_mat_unique,n_obj\n")
            for j in range(len(cams)):
                f.write("%f,%f,%f,%s,%i,%i,%i,%i\n" % \
                        (acc_ind[j], cams[j][0], cams[j][1], tfs[j], nc[j], ns[j], nm[j], n_obj[j]))
