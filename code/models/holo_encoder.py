import torch
from collections import OrderedDict
from torch import optim
from torch.nn import functional as F
from itertools import chain
from .base import Base
from .holo_ae import HoloAE
from torch import nn

class HoloEncoder(Base):
    """
    """

    def __init__(self,
                 enc,
                 probe,
                 disable_rot=False,
                 rot_consist=0.,
                 cls_loss='cce',
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002,
                           'betas': (0.5, 0.999)},
                 handlers=[]):
        super(HoloEncoder, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.enc = enc
        self.probe = probe
        self.disable_rot = disable_rot
        self.rot_consist = rot_consist

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.enc.to(self.device)
            self.probe.to(self.device)

        if cls_loss == 'cce':
            self.cls_loss_fn = nn.CrossEntropyLoss()
        elif cls_loss == 'mse':
            self.cls_loss_fn = self.mse
        else:
            raise Exception("Only cce or mse is currently supported for cls_loss")

        self.optim = {
            'g': opt(
                filter(lambda p: p.requires_grad, enc.parameters()),
            **opt_args)
        }
        optim_probe = opt(
            filter(lambda p: p.requires_grad, probe.parameters()),
            **opt_args)
        self.optim['probe'] = optim_probe

        for elem in self.optim.values():
            print(elem)

        self.schedulers = []
        self.handlers = handlers

        self.last_epoch = 0
        self.load_strict = True

    def _train(self):
        self.enc.train()
        self.probe.train()

    def train(self):
        self._train()

    def _eval(self):
        self.enc.eval()
        self.probe.eval()

    def eval(self):
        self._eval()

    def mse(self, prediction, target):
        return torch.mean((prediction-target)**2)

    def t_rot_matrix_x(self, theta):
        """
        theta: measured in radians
        """
        bs = theta.size(0)
        mat = torch.zeros((bs, 3, 3)).float()
        if theta.is_cuda:
            mat = mat.cuda()
        mat[:, 0, 0] = 1.
        mat[:, 1, 1] = torch.cos(theta).view(-1)
        mat[:, 1, 2] = -torch.sin(theta).view(-1)
        mat[:, 2, 1] = torch.sin(theta).view(-1)
        mat[:, 2, 2] = torch.cos(theta).view(-1)
        return mat

    def t_rot_matrix_y(self, theta):
        """
        theta: measured in radians
        """
        bs = theta.size(0)
        mat = torch.zeros((bs, 3, 3)).float()
        if theta.is_cuda:
            mat = mat.cuda()
        mat[:, 0, 0] = torch.cos(theta).view(-1)
        mat[:, 0, 2] = torch.sin(theta).view(-1)
        mat[:, 1, 1] = 1.
        mat[:, 2, 0] = -torch.sin(theta).view(-1)
        mat[:, 2, 2] = torch.cos(theta).view(-1)
        return mat

    def t_rot_matrix_z(self, theta):
        """
        theta: measured in radians
        """
        bs = theta.size(0)
        mat = torch.zeros((bs, 3, 3)).float()
        if theta.is_cuda:
            mat = mat.cuda()
        mat[:, 0, 0] = torch.cos(theta).view(-1)
        mat[:, 0, 1] = -torch.sin(theta).view(-1)
        mat[:, 1, 0] = torch.sin(theta).view(-1)
        mat[:, 1, 1] = torch.cos(theta).view(-1)
        mat[:, 2, 2] = 1.
        return mat

    def t_get_theta(self, angles, offsets=None):
        '''Construct a rotation matrix from angles. (This is
        the differentiable version, in PyTorch code.)

        angles should be an nx3 matrix with z,y,z = 0,1,2
        '''

        #angles_x = angles[:, 0]
        #angles_z = angles[:, 1]
        #angles_y = angles[:, 2]

        angles_z = angles[:, 0]
        angles_y = angles[:, 1]
        angles_x = angles[:, 2]

        if offsets is not None:
            trans_x = offsets[:, 0]
            trans_y = offsets[:, 1]
            trans_z = offsets[:, 2]

        thetas = torch.bmm(torch.bmm(self.t_rot_matrix_z(angles_z),
                                     self.t_rot_matrix_y(angles_y)),
                           self.t_rot_matrix_x(angles_x))
        trans = torch.zeros((thetas.size(0), 3, 1))

        if offsets is not None:
            trans[:, 0, :] = trans_x.view(-1, 1)
            trans[:, 1, :] = trans_y.view(-1, 1)
            trans[:, 2, :] = trans_z.view(-1, 1)
        if angles.is_cuda:
            trans = trans.cuda()
        thetas = torch.cat((thetas, trans), dim=2) # add zero padding
        return thetas

    def stn(self, x, theta):
        # theta must be (Bs, 3, 4) = [R|t]
        #theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        if x.is_cuda:
            grid = grid.to(self.device)
        out = F.grid_sample(x, grid, padding_mode='border')
        return out

    def _pad(self, x):
        zeros = torch.zeros((x.size(0), 1, 4)).cuda()
        zeros[:, :, -1] = 1.
        x_pad = torch.cat((x, zeros), dim=1)
        return x_pad

    def train_on_instance(self,
                          x_batch,
                          x2_batch,
                          q_batch,
                          cam_batch,
                          cam2_batch,
                          y_batch,
                          meta_batch,
                          **kwargs):
        self._train()
        for key in self.optim:
            self.optim[key].zero_grad()

        losses = {}

        z = self.enc.encode(x_batch)

        if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
            print("Debugging info:")
            print("  x shape:", x_batch.shape)
            print("  cam shape:", cam_batch.shape)
            print("  z shape:", z.shape)
        h = self.enc.enc2vol(z) # 'template canonical'

        if not self.disable_rot:
            theta = self.enc.cam_encode(cam_batch)
            rot_mat = self.t_get_theta(theta[:, 0:3], theta[:, 3:6])
            h_rot = self.stn(h, rot_mat) # actual viewpoint
        else:
            h_rot = h

        if self.rot_consist > 0:
            if self.disable_rot:
                raise Exception("rot_consist only works if disable_rot=False")

            # We are given the camera coordinates of both
            # x1 and x2.
            #
            # cam_relate(cam1, cam2) = rot matrix s.t.
            # rot(h1, rot_matrix) == h2.

            h2 = self.enc.enc2vol(self.enc.encode(x2_batch))

            # Map h1 to h2
            thetas_12 = self.enc.cam_infer(cam_batch, cam2_batch)
            R_12 = self.t_get_theta(thetas_12[:, 0:3], thetas_12[:, 3:6])
            h2_pred = self.stn(h, R_12)
            h12_consist = torch.mean(torch.abs(h2_pred-h2))

            # Make sure that cam_infer(cam2, cam1) ==
            # the inverse of R_21
            R_21_inv = torch.inverse(self._pad(R_12))[:, 0:3]
            thetas_21 = self.enc.cam_infer(cam2_batch, cam_batch)
            R_21 = self.t_get_theta(thetas_21[:, 0:3], thetas_21[:, 3:6])
            h_inv_consist = torch.mean(torch.abs(R_21-R_21_inv))

            # Also handle identity loss
            thetas_11 = self.enc.cam_infer(cam_batch, cam_batch)
            h_blank = torch.mean((thetas_11)**2)

            h_consist = h12_consist + \
                h_inv_consist + \
                h_blank

        if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
            print("  h shape:", h.shape)

        #z = self.enc.encode(x_batch)
        #h_rot = z
        if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
            print("Debugging info:")
            print("  z shape:", z.shape)

        #z = self.enc.encode(x_batch)
        #if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
        #    print("Debugging info:")
        #    print("  z shape:", z.shape)

        probe_out = self.probe(h_rot, q_batch, cam_batch)
        probe_loss = self.cls_loss_fn(probe_out, y_batch)
        if self.rot_consist > 0:
            probe_loss = probe_loss + self.rot_consist*h_consist
        probe_loss.backward()
        with torch.no_grad():
            if self.cls_loss_fn != self.mse:
                probe_acc = (probe_out.argmax(dim=1).long() == y_batch).float().mean()
            else:
                probe_acc = probe_loss

        self.optim['g'].step()
        self.optim['probe'].step()

        losses['probe_loss'] = probe_loss.item()
        losses['probe_acc'] = probe_acc.item()

        if self.rot_consist > 0:
            losses['rot_consist'] = h_consist.item()
            #losses['theta1_norm'] = (theta**2).mean().item()
            #losses['theta2_norm'] = (theta2**2).mean().item()

        #with torch.no_grad():
            # Want to know the variation in
            # outputted thetas.
        #    losses['theta_std'] = theta.std().item()

        outputs = {
        }

        return losses, outputs

    def eval_on_instance(self,
                         x_batch,
                         x2_batch,
                         q_batch,
                         cam_batch,
                         cam2_batch,
                         y_batch,
                         meta_batch,
                         **kwargs):
        self._eval()
        losses = {}
        with torch.no_grad():


            z = self.enc.encode(x_batch)
            h = self.enc.enc2vol(z) # 'template canonical'
            if not self.disable_rot:
                theta = self.enc.cam_encode(cam_batch)
                rot_mat = self.t_get_theta(theta[:, 0:3], theta[:, 3:6])
                h_rot = self.stn(h, rot_mat) # actual viewpoint
            else:
                h_rot = h

            #z = self.enc.encode(x_batch)
            #h_rot = z

            probe_out = self.probe(h_rot, q_batch, cam_batch)
            probe_loss = self.cls_loss_fn(probe_out, y_batch)
            if self.cls_loss_fn != self.mse:
                probe_acc = (probe_out.argmax(dim=1).long() == y_batch).float().mean()
            else:
                probe_acc = probe_loss
            losses['probe_loss'] = probe_loss.item()
            losses['probe_acc'] = probe_acc.item()

        return losses, {}

    def predict(self,
                x_batch,
                x2_batch,
                q_batch,
                cam_batch,
                cam2_batch,
                y_batch=None,
                meta_batch=None,
                **kwargs):
        self._eval()
        with torch.no_grad():
            z = self.enc.encode(x_batch)
            h = self.enc.enc2vol(z) # 'template canonical'
            if not self.disable_rot:
                theta = self.enc.cam_encode(cam_batch)
                rot_mat = self.t_get_theta(theta[:, 0:3], theta[:, 3:6])
                h_rot = self.stn(h, rot_mat) # actual viewpoint
            else:
                h_rot = h

            probe_out = self.probe(h_rot, q_batch, cam_batch)
            return probe_out

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
        dd['enc'] = self.enc.state_dict()
        if self.probe is not None:
            dd['probe'] = self.probe.state_dict()
        # Save the models' optim state.
        for key in self.optim:
            dd['optim_%s' % key] = self.optim[key].state_dict()
        dd['epoch'] = epoch
        torch.save(dd, filename)

    def load(self, filename):
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        dd = torch.load(filename,
                        map_location=map_location)

        # Load the models.
        self.enc.load_state_dict(dd['enc'],
                                 strict=self.load_strict)
        if self.probe is not None:
            self.probe.load_state_dict(dd['probe'],
                                       strict=self.load_strict)
        # Load the models' optim state.
        try:
            for key in self.optim:
                if ('optim_%s' % key) in dd:
                    self.optim[key].load_state_dict(dd['optim_%s' % key])
        except:
            print("WARNING: was unable to load state dict for optim")
            print("This is not a big deal if you're only using " + \
                  "the model for inference, however.")
        self.last_epoch = dd['epoch']

