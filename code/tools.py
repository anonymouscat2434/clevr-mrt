import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
import sys
import tempfile
import glob
from collections import OrderedDict
from torchvision.utils import save_image
from tqdm import tqdm

rot2idx = {
    'yaw': 0,
    'pitch': 1,
    'roll': 2
}


def find_latest_pkl_in_folder(model_dir):
    # List all the pkl files.
    files = glob.glob("%s/*.pkl" % model_dir)
    # Make them absolute paths.
    files = [os.path.abspath(key) for key in files]
    if len(files) > 0:
        # Get creation time and use that.
        latest_model = max(files, key=os.path.getctime)
        print("Auto-resume mode found latest model: %s" %
              latest_model)
        return latest_model
    return None

def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

def generate_name_from_args(dd, kwargs_for_name):
    buf = {}
    for key in dd:
        if key in kwargs_for_name:
            if dd[key] is None:
                continue
            new_name, fn_to_apply = kwargs_for_name[key]
            new_val = fn_to_apply(dd[key])
            if dd[key] is True:
                new_val = ''
            buf[new_name] = new_val
    buf_sorted = OrderedDict(sorted(buf.items()))
    #tags = sorted(tags.split(","))
    name = ",".join([ "%s=%s" % (key, buf_sorted[key]) for key in buf_sorted.keys()])
    return name

def line2dict(st):
    """Convert a line of key=value pairs to a
    dictionary.

    :param st:
    :returns: a dictionary
    :rtype:
    """
    if st is None:
        return {}
    elems = st.split(';')
    dd = {}
    for elem in elems:
        elem = elem.split('=')
        key, val = elem
        if is_int(val):
            dd[key] = int(val)
        elif is_float(val):
            dd[key] = float(val)
        elif val.lower() == 'false':
            dd[key] = False
        elif val.lower() == 'true':
            dd[key] = True
        else:
            dd[key] = val
    return dd

def count_params(module, trainable_only=True):
    """Count the number of parameters in a
    module.

    :param module: PyTorch module
    :param trainable_only: only count trainable
      parameters.
    :returns: number of parameters
    :rtype:

    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num

def save_rotations(gan,
                   x_batch,
                   out_folder,
                   angle_override=None,
                   num=10,
                   nrow=10,
                   min_angle=-2.,
                   max_angle=2.,
                   num_repeats=1,
                   rsf=1.,
                   rot_mask=None,
                   offset_mask=None,
                   padding=2):
    """Save interpolations between a batch and its permuted
    version to disk
    :param gan:
    :param x_batch:
    :param out_folder:
    :param num: number of interpolation steps to perform
    :param mix_input: if `True`, only produce input space mix
    :param padding: padding on image grid
    :returns:
    :rtype:
    """
    gan._eval()

    if rot_mask is not None:
        gan.rz, gan.ry, gan.rx = rot_mask
    if offset_mask is not None:
        gan.tx, gan.ty, gan.tz = offset_mask

    # Add a subfolder saying what the min and max angles are.
    #out_folder = "%s/%f_%f" % (out_folder, gan.min_angle, gan.max_angle)
    out_folder = "%s/preset" % (out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with torch.no_grad():

        h, (rz, ry, rx, tx, ty, tz) = gan.encode(x_batch)

        for k in range(num_repeats):

            for axis in ['x', 'z', 'y']:
                pbuf = []
                for p in np.linspace(
                        min_angle*rsf,
                        max_angle*rsf,
                        num=num):
                    if axis == 'x':
                        h_rot = gan.rotate(h,
                                           rz[1]*0, ry[1]*0, rx[1]*0 + p,
                                           tx[1]*0, ty[1]*0, tz[1]*0)
                    elif axis == 'z':
                        h_rot = gan.rotate(h,
                                           rz[1]*0 + p, ry[1]*0, rx[1]*0,
                                           tx[1]*0, ty[1]*0, tz[1]*0)
                    elif axis == 'y':
                        h_rot = gan.rotate(h,
                                           rz[1]*0, ry[1]*0 + p, rx[1]*0,
                                           tx[1]*0, ty[1]*0, tz[1]*0)

                    dec_enc_rot = gan.generator.decode(h_rot)

                    pbuf.append(dec_enc_rot.detach().cpu())

                for b in range(x_batch.size(0)):
                    this_interp = torch.stack([pbuf[i][b] for i in range(len(pbuf))])
                    out_file = "{folder}/{k}_{axis}_b{b}.png".format(
                        folder=out_folder,
                        axis=axis,
                        b=b,
                        k=k
                    )
                    save_image( this_interp*0.5 + 0.5,
                                nrow=nrow,
                                filename=out_file,
                                padding=padding,
                                pad_value=1)

                # Also save a version where all the rows are in one image.
                pbuf_all = torch.stack(pbuf, dim=0).transpose(0, 1).contiguous()
                pbuf_all = pbuf_all.view(-1, pbuf[0].size(1), pbuf[0].size(2), pbuf[0].size(3))

                out_file = "{folder}/{k}_{axis}_all.png".format(
                    folder=out_folder,
                    k=k,
                    axis=axis,
                )
                save_image(pbuf_all*0.5 + 0.5, nrow=len(pbuf),
                           filename=out_file,
                           pad_value=1)

def save_recon(gan,
               x_batch,
               out_folder,
               angle_override=None,
               n_samples=10,
               std_multiplier=1.0,
               rot_mask=None,
               offset_mask=None,
               padding=2):
    """Save interpolations between a batch and its permuted
    version to disk
    :param gan:
    :param x_batch:
    :param out_folder:
    :param num: number of interpolation steps to perform
    :param mix_input: if `True`, only produce input space mix
    :param padding: padding on image grid
    :returns:
    :rtype:
    """
    gan._eval()

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with torch.no_grad():

        # We have to infer the offsets and angles, and then
        # override the one which we are interpolating over.
        enc_mu, enc_std, q = gan.generator.encode(x_batch)
        rz, ry, rx, tx, ty, tz = gan._extract_angles_and_offsets(q)
        enc_distn = gan.sample_z(enc_mu, torch.exp(enc_std)*std_multiplier)[0]

        print("Inferred mean of z")
        print(enc_mu.mean())
        print("Inferred std of z:")
        print(torch.exp(enc_std).mean())
        print("Inferred std of z with multiplier %f:" % std_multiplier)
        print(torch.exp(enc_std).mean()*std_multiplier)

        # Produce reconstructions with different w's.
        pbuf = []
        pbuf_null = []
        for j in range(n_samples):

            h = gan.generator.enc2vol(enc_distn.rsample())
            h_rot = gan.rotate(h,
                               rz[1], ry[1], rx[1],
                               tx[1], ty[1], tz[1])
            dec_enc_rot = gan.generator.decode(h_rot)
            pbuf.append(dec_enc_rot.detach().cpu())

            dec_enc_null = gan.generator.decode(h)
            pbuf_null.append(dec_enc_null.detach().cpu())

        # Also save a version where all the rows are in one image.
        pbuf_all = torch.stack(pbuf, dim=0).transpose(0, 1).contiguous()
        pbuf_all = pbuf_all.view(-1, pbuf[0].size(1), pbuf[0].size(2), pbuf[0].size(3))

        out_file = "{folder}/recon.png".format(
            folder=out_folder
        )
        save_image(pbuf_all*0.5 + 0.5, nrow=len(pbuf),
                   filename=out_file,
                   pad_value=1)

        pbuf_null_all = torch.stack(pbuf_null, dim=0).transpose(0, 1).contiguous()
        pbuf_null_all = pbuf_null_all.view(-1, pbuf_null[0].size(1),
                                           pbuf_null[0].size(2),
                                           pbuf_null[0].size(3))

        out_file = "{folder}/recon_null.png".format(
            folder=out_folder
        )
        save_image(pbuf_null_all*0.5 + 0.5, nrow=len(pbuf),
                   filename=out_file,
                   pad_value=1)



def save_translation(gan,
                     x_batch,
                     out_folder,
                     angle_override=None,
                     num=10,
                     nrow=10,
                     padding=2,
                     rot_mask=None,
                     offset_mask=None,
                     zero_input=False):
    """Test out translations.
    :param gan:
    :param x_batch:
    :param out_folder:
    :param num: number of interpolation steps to perform
    :param mix_input: if `True`, only produce input space mix
    :param padding: padding on image grid
    :returns:
    :rtype:
    """
    gan._eval()

    if rot_mask is not None:
        gan.rz, gan.ry, gan.rx = rot_mask
    if offset_mask is not None:
        gan.tx, gan.ty, gan.tz = offset_mask

    with torch.no_grad():

        # We have to infer the offsets and angles, and then
        # override the one which we are interpolating over.
        enc, q, h = gan.generator.encode(x_batch)
        rz, ry, rx, tx, ty, tz = gan._extract_angles_and_offsets(q)
        #angles = torch.cat((rx[1]*gan.cx, rz[1]*gan.cz, ry[1]*gan.cy), dim=1)
        #offsets = torch.cat((tx[1], ty[1], tz[1]), dim=1)

        for b, axis in enumerate(['x', 'y', 'z']):
            pbuf = []
            # Add a subfolder saying what the min and max angles are.
            this_out_folder = "%s/%s" % (out_folder, axis)
            if not os.path.exists(this_out_folder):
                os.makedirs(this_out_folder)

            for p in np.linspace(-0.5, 0.5, num=num):
                if axis == 'x':
                    h_rot = gan.rotate(h,
                                       rz[1], ry[1], rx[1],
                                       tx[1]*0 + p, ty[1], tz[1])
                elif axis == 'z':
                    h_rot = gan.rotate(h,
                                       rz[1], ry[1], rx[1],
                                       tx[1], ty[1]*0 + p, tz[1])
                elif axis == 'y':
                    h_rot = gan.rotate(h,
                                       rz[1], ry[1], rx[1],
                                       tx[1], ty[1], tz[1]*0 + p)

                dec_enc_rot = gan.generator.decode(h_rot, enc)
                pbuf.append(dec_enc_rot.detach().cpu())

            # Also save a version where all the rows are in one image.
            pbuf_all = torch.stack(pbuf, dim=0).transpose(0, 1).contiguous()
            pbuf_all = pbuf_all.view(-1, pbuf[0].size(1), pbuf[0].size(2), pbuf[0].size(3))
            save_image(pbuf_all*0.5 + 0.5, nrow=len(pbuf),
                       filename="%s/all_%s.png" % (this_out_folder, axis),
                       pad_value=1)

def save_mix(gan,
             x_batch,
             x2_batch,
             out_folder,
             angle_override=None,
             num=10,
             nrow=10,
             padding=2):
    """Test out mixup.
    :param gan:
    :param x_batch:
    :param out_folder:
    :param num: number of interpolation steps to perform
    :param mix_input: if `True`, only produce input space mix
    :param padding: padding on image grid
    :returns:
    :rtype:
    """
    #gan._eval()

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # BUG: this damn thing don't work
    # in eval mode.
    gan.generator.train()

    with torch.no_grad():



        enc, q = gan.generator.encode(x_batch, x2_batch)
        rz, ry, rx, tx, ty, tz = gan._extract_angles_and_offsets(q)

        h1 = gan.generator.enc2vol(enc)

        h1_onto_h2 = gan.rotate(h1,
                                rz[1], ry[1], rx[1],
                                tx[1], ty[1], tz[1])
        x_h1_onto_h2 = gan.generator.decode(h1_onto_h2)
        x_h1 = gan.generator.decode(h1)

        save_image(x_h1*0.5 + 0.5,
                   filename="%s/x1.png" % (out_folder),
                   pad_value=1)
        save_image(x_h1_onto_h2*0.5 + 0.5,
                   filename="%s/x1_to_x2.png" % (out_folder),
                   pad_value=1)

        save_image(x_batch*0.5 + 0.5,
                   filename="%s/x1_real.png" % (out_folder),
                   pad_value=1)
        save_image(x2_batch*0.5 + 0.5,
                   filename="%s/x2_real.png" % (out_folder),
                   pad_value=1)


        # Interpolate from identity (origin) to the
        # predicted rotation to x2.
        pbuf = []
        for alpha in np.linspace(0, 1, num=num):
            _rz, _ry, _rx, _tx, _ty, _tz = \
                rz[1]*alpha, ry[1]*alpha, rx[1]*alpha, \
                tx[1]*alpha, ty[1]*alpha, tz[1]*alpha
            print(_rz)
            h1_interp = gan.rotate(h1,
                                   _rz, _ry, _rx,
                                   _tx, _ty, _tz)
            x_onto_interp = gan.generator.decode(h1_interp)
            pbuf.append(x_onto_interp.detach().cpu())

        pbuf_all = torch.stack(pbuf, dim=0).transpose(0, 1).contiguous()
        pbuf_all = pbuf_all.view(-1, pbuf[0].size(1), pbuf[0].size(2), pbuf[0].size(3))
        save_image(pbuf_all*0.5 + 0.5, nrow=len(pbuf),
                   filename="%s/interp.png" % (out_folder),
                   pad_value=1)


def save_frames(gan,
                x_batch,
                out_folder,
                num=10,
                rsf=1.0,
                crf=23,
                fps=24,
                upscale=1,
                min_angle=-2,
                max_angle=2,
                zero_input=False):

    from subprocess import check_output

    gan._eval()
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    tmp_dir = tempfile.mkdtemp()
    print("Temp dir: %s" % tmp_dir)

    with torch.no_grad():

        # Undo the rotation to put the h
        # into canonical pose.
        h, (rz, ry, rx, tx, ty, tz) = \
            gan.encode(x_batch)

        for axis in ['x', 'z', 'y', 'tx', 'ty', 'tz']:

            this_folder = "%s/%s" % (tmp_dir, axis)
            if not os.path.exists(this_folder):
                os.makedirs(this_folder)

            print("Generating frames for axis %s..." % axis)
            for interp_idx, p in enumerate(
                    np.linspace(
                        min_angle*rsf,
                        max_angle*rsf,
                        num=num)):
                if axis == 'x':
                    h_rot = gan.rotate(h,
                                       rz[1]*0, ry[1]*0, rx[1]*0 + p,
                                       tx[1]*0, ty[1]*0, tz[1]*0)
                elif axis == 'z':
                    h_rot = gan.rotate(h,
                                       rz[1]*0 + p, ry[1]*0, rx[1]*0,
                                       tx[1]*0, ty[1]*0, tz[1]*0)
                elif axis == 'y':
                    h_rot = gan.rotate(h,
                                       rz[1]*0, ry[1]*0 + p, rx[1]*0,
                                       tx[1]*0, ty[1]*0, tz[1]*0)
                elif axis == 'tx':
                    h_rot = gan.rotate(h,
                                       rz[1]*0, ry[1]*0, rx[1]*0,
                                       tx[1]*0 + p, ty[1], tz[1]*0)
                elif axis == 'ty':
                    h_rot = gan.rotate(h,
                                       rz[1]*0, ry[1]*0, rx[1]*0,
                                       tx[1]*0, ty[1]*0 + p, tz[1]*0)
                elif axis == 'tz':
                    h_rot = gan.rotate(h,
                                       rz[1]*0, ry[1]*0, rx[1]*0,
                                       tx[1]*0, ty[1]*0, tz[1]*0 + p)

                dec_enc_rot = gan.generator.decode(h_rot)
                if upscale > 1:
                    dec_enc_rot = F.interpolate(dec_enc_rot,
                                                scale_factor=upscale,
                                                mode='bilinear')

                out_file = "%s/%s/{0:06d}.png".format(interp_idx) % (tmp_dir, axis)
                save_image(dec_enc_rot*0.5 + 0.5,
                           filename=out_file,
                           padding=0)

            this_out_folder = "%s/%s" % (out_folder, axis)
            this_in_folder = "%s/%s" % (tmp_dir, axis)
            for f_ in [this_out_folder, this_in_folder]:
                if not os.path.exists(f_):
                    os.makedirs(f_)
            # Remove old mp4 file if it exists.
            if os.path.exists("%s/out.mp4" % this_out_folder):
                os.remove("%s/out.mp4" % this_out_folder)

            print("this_in_folder", this_in_folder)
            print("this_out_folder", this_out_folder)

            ffmpeg_out = check_output(
                "cd %s; ffmpeg -framerate %i -pattern_type glob -i '*.png' -crf %i -c:v libx264 out.mp4" % (this_in_folder, fps, crf),
                shell=True)
            ffmpeg_out = ffmpeg_out.decode('utf-8').rstrip()
            print(ffmpeg_out)

            copy_out = check_output(
                "cp %s/out.mp4 %s/out.mp4" % (this_in_folder, this_out_folder),
                shell=True
            )
            print(copy_out)

def compute_fid(loader,
                gan,
                cls,
                max_samples=10000,
                num_repeats=5):

    from fid_score import calculate_fid_given_imgs

    # Collect the training set.
    train_samples = []
    gen_samples = []
    for b, (x_batch, _) in enumerate(loader):
        if b*loader.batch_size >= max_samples:
            break
        train_samples.append( (((x_batch.numpy()*0.5) + 0.5)*255.).astype(np.int32) )
    train_samples = np.vstack(train_samples)

    #########################################
    # Write FID between samples and dataset #
    #########################################

    with torch.no_grad():

        use_cuda = gan.use_cuda
        scores = []
        for iter_ in range(num_repeats):

            gen_samples = []
            for b, (x_batch, _) in enumerate(loader):
                if b*loader.batch_size >= max_samples:
                    break
                this_sample = gan.sample(x_batch).cpu().numpy()
                this_sample = (((this_sample*0.5) + 0.5)*255.).astype(np.int32)
                gen_samples.append(this_sample)

            gen_samples = np.vstack(gen_samples)

            score = calculate_fid_given_imgs(train_samples,
                                             gen_samples,
                                             16,
                                             use_cuda,
                                             dims=512,
                                             model=cls)
            scores.append(score)
        return np.mean(scores)

def extract_angles(gan, loader, out_dir):
    """
    """
    gan._eval()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with torch.no_grad():
        arr_rx = []
        arr_rz = []
        arr_ry = []
        arr_tx = []
        arr_ty = []
        arr_tz = []
        pbar = tqdm(total=len(loader))
        for idx, batch in enumerate(loader):
            batch = gan.prepare_batch(batch)
            enc, q, h = gan.generator.encode(batch[0])
            rz, ry, rx, tx, ty, tz = gan._extract_angles_and_offsets(q)
            arr_rx.append(rx[1])
            arr_rz.append(rz[1])
            arr_ry.append(ry[1])
            arr_tx.append(tx[1])
            arr_ty.append(ty[1])
            arr_tz.append(tz[1])
            pbar.update(1)
    arr_rx = torch.cat(arr_rx, dim=0).flatten().cpu().numpy()
    arr_rz = torch.cat(arr_rz, dim=0).flatten().cpu().numpy()
    arr_ry = torch.cat(arr_ry, dim=0).flatten().cpu().numpy()
    arr_tx = torch.cat(arr_tx, dim=0).flatten().cpu().numpy()
    arr_ty = torch.cat(arr_ty, dim=0).flatten().cpu().numpy()
    arr_tz = torch.cat(arr_tz, dim=0).flatten().cpu().numpy()

    print("rx:", arr_rx.mean(), arr_rx.std())
    print("rz:", arr_rz.mean(), arr_rz.std())
    print("ry:", arr_ry.mean(), arr_ry.std())
    print("tx:", arr_tx.mean(), arr_tx.std())
    print("ty:", arr_ty.mean(), arr_ty.std())
    print("tz:", arr_tz.mean(), arr_tz.std())

    print("Saving to: %s" % out_dir)
    np.savez(file="%s/stats.npz" % out_dir,
             rx=arr_rx,
             rz=arr_rz,
             ry=arr_ry,
             tx=arr_tx,
             ty=arr_ty,
             tz=arr_tz)
