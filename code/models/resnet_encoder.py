import torch
from collections import OrderedDict
from torch import optim
from torch.nn import functional as F
from itertools import chain
from .base import Base
from .holo_ae import HoloAE
from torch import nn

class ResnetEncoder(Base):
    """
    Intended to be used as a FILM baseline.
    """

    def __init__(self,
                 enc,
                 probe,
                 disable_rot=False,
                 rot_consist=False,
                 cls_loss='cce',
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002,
                           'betas': (0.5, 0.999)},
                 handlers=[]):
        super(ResnetEncoder, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("device:", self.device)

        self.enc = enc
        self.probe = probe

        self.use_cuda = use_cuda
        if self.use_cuda:
            if self.probe is not None:
                self.probe.to(self.device)

        if cls_loss == 'cce':
            self.cls_loss_fn = nn.CrossEntropyLoss()
        elif cls_loss == 'mse':
            self.cls_loss_fn = self.mse
        else:
            raise Exception("Only cce or mse is currently supported for cls_loss")

        self.optim = {}
        optim_probe = opt(filter(lambda p: p.requires_grad, probe.parameters()), **opt_args)
        self.optim['probe'] = optim_probe

        for elem in self.optim.values():
            print(elem)

        self.schedulers = []
        self.handlers = handlers

        self.last_epoch = 0
        self.load_strict = True

    def _train(self):
        # Encoder is not trained, it is always
        # in eval mode.
        self.enc.eval()
        if self.probe is not None:
            self.probe.train()

    def _eval(self):
        self.enc.eval()
        if self.probe is not None:
            self.probe.eval()

    def mse(self, prediction, target):
        return torch.mean((prediction-target)**2)

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

        if hasattr(self.enc, 'encode'):
            enc = self.enc.encode(x_batch)[0].detach()
        else:
            # Must be imagenet model
            enc = self.enc(x_batch).detach()

        if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
            print("Debugging info:")
            print("  x shape:", x_batch.shape)
            print("  enc shape:", enc.shape)
            print("  cam shape:", cam_batch.shape)

        probe_out = self.probe(enc, q_batch, cam_batch)
        probe_loss = self.cls_loss_fn(probe_out, y_batch)
        probe_loss.backward()
        with torch.no_grad():
            if self.cls_loss_fn != self.mse:
                probe_acc = (probe_out.argmax(dim=1).long() == y_batch).float().mean()
            else:
                probe_acc = probe_loss
        self.optim['probe'].step()

        losses = {}
        losses['probe_loss'] = probe_loss.item()
        losses['probe_acc'] = probe_acc.item()

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

            if hasattr(self.enc, 'encode'):
                enc = self.enc.encode(x_batch)[0].detach()
            else:
                # Must be imagenet model
                enc = self.enc(x_batch).detach()

            probe_out = self.probe(enc, q_batch, cam_batch)
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
            if hasattr(self.enc, 'encode'):
                enc = self.enc.encode(x_batch)[0].detach()
            else:
                # Must be imagenet model
                enc = self.enc(x_batch).detach()

            probe_out = self.probe(enc, q_batch, cam_batch)
            return probe_out

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
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
        self.probe.load_state_dict(dd['probe'], strict=self.load_strict)
        # Load the models' optim state.
        for key in self.optim:
            if ('optim_%s' % key) in dd:
                self.optim[key].load_state_dict(dd['optim_%s' % key])
        self.last_epoch = dd['epoch']

