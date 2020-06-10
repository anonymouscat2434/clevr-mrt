# Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/stargan/datasets.py

import glob
import random
import os
import numpy as np
import torch
from scipy import io
import pickle
#import h5py
import torch.nn.functional as F
import json
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
import h5py

from .utils import load_vocab

class BlankDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError()
    def __len__(self):
        return 1

class ClevrKiwiDataset(Dataset):
    def __init__(self,
                 root_images,
                 root_meta,
                 transforms_=None,
                 mode='train',
                 canonical_mode='train_only',
                 img_size=128):

        if canonical_mode not in ['b0', 'none', 'train_only']:
            # b0 = canonical view only (like vanilla clevr)
            # none = no canonical views at all
            # train_only = canonical views included in train, eval no
            raise Exception("bad canonical mode")

        subfolders = glob.glob("%s/*" % root_images)

        self.root_images = root_images
        self.root_meta = root_meta
        self.transform = transforms.Compose(transforms_)
        self.canonical_mode = canonical_mode

        self.vocab = load_vocab(os.environ["DATASET_CLEVR_KIWI_META"] +
                                "/vocab.json")

        if mode not in ['train', 'val', 'test']:
            raise Exception("mode must be either train or val or test (got %s)" % mode)
        self.mode = mode

        # This holds every question and for all intents
        # and purposes is the _length_ of this dataset.
        # In order to map a question to its scene we
        # must parse its filename and use id_to_scene
        # in order to go from question to camera views.
        if mode == 'train':
            h5 = h5py.File("%s/train_questions.h5" % root_meta, "r")
        elif mode == 'val':
            h5 = h5py.File("%s/valid_questions.h5" % root_meta, "r")
        else:
            h5 = h5py.File("%s/test_questions.h5" % root_meta, "r")

        self.answers = h5['answers'][:]
        self.image_filenames = [ x.decode('utf-8') for x in h5['image_filenames'][:] ]
        self.template_filenames = [x.decode('utf-8') for x in h5['template_filenames'][:] ]
        self.questions = h5['questions'][:]
        self.question_strs = h5['question_strs'][:]

        assert len(self.answers) == len(self.image_filenames) == len(self.questions)

        if mode in ['train', 'val']:
            cache_file = "%s/cache.pkl" % root_meta
        else:
            cache_file = "%s/cache_test.pkl" % root_meta

        if not os.path.exists(cache_file):

            print("Cannot find %s, so generating it..." % cache_file)

            id_to_scene = {}

            n_questions = 0
            for subfolder in subfolders:
                q_file = "%s/questions.json" % subfolder
                s_file = "%s/scenes.json" % subfolder

                if not os.path.exists(q_file) or not os.path.exists(s_file):
                    print("ERROR: skip:", subfolder)
                    continue

                q_json = json.loads(open(q_file).read())
                s_json = json.loads(open(s_file).read())

                n_questions += len(q_json['questions'])

                # Collect scenes first.
                for idx, scene in enumerate(s_json['scenes']):
                    # Add subfolder to scene dict
                    for key in scene:
                        scene[key]['subfolder'] = os.path.basename(subfolder)

                    this_scene_cc = scene['cc']
                    # e.g. 's002400'
                    this_basename = this_scene_cc['image_filename'].split("_")[-2]
                    # Map the basename e.g. s002400
                    # to its dictionary of camera views.
                    id_to_scene[this_basename] = scene

            self.id_to_scene = id_to_scene

            print("Writing cache to: %s" % root_meta)
            print("(NOTE: if you change the dataset, delete the cache file")
            with open(cache_file, "wb") as f_write:
                pickle.dump(
                    self.id_to_scene,
                    f_write
                )

        else:

            with open(cache_file, "rb") as f_read:
                print("Loading cache file...")
                self.id_to_scene = pickle.load(f_read)

        self.mode = mode
        # TODO: cleanup
        self.cam_names = ['cam1', 'cam5', 'cam19', 'cam7', 'cam3', 'cam16', 'cam18', 'cam6', 'cam14', 'cam17', 'cam13', 'cam8', 'cc', 'cam2', 'cam15', 'cam10', 'cam12', 'cam0', 'cam4', 'cam9', 'cam11']

        print("DEBUGGING INFO:")
        print("  # of questions:", len(self.questions)) # 14922
        # self.image_filename is all cc's, but let it denote the 'scene'
        print("  # of unique scenes:", len(set(self.image_filenames))) # 2992

        self.img_size = img_size

    def __getitem__(self, index):
        # Ok, grab the metadata
        this_q = torch.from_numpy(self.questions[index]).long()
        this_answer = torch.LongTensor([self.answers[index]])
        this_filename_cc = self.image_filenames[index]
        this_id = this_filename_cc.split("_")[-2]
        #this_q_family = self.question_family[index]
        this_template_filename = self.template_filenames[index]

        # DEBUG
        this_q_str = self.question_strs[index]
        recon_q = [ self.vocab['question_idx_to_token'][c.item()] for c in this_q ]

        # np.where(this_q == 0)[0][0]
        # len(this_q_str.decode('utf-8').split(" "))

        # A dictionary of keys consisting of camera
        # views.
        scene_from_id = self.id_to_scene[this_id]

        subfolder = scene_from_id['cc']['subfolder']

        cam_names = self.cam_names
        # If validation set, don't use a canonical
        # pose image (to be more difficult)
        if self.canonical_mode == 'train_only':
            # Canonical view is only meant to be
            # in train, so if this is valid, remove
            # 'cc' from the array.
            if self.mode == 'val':
                cam_names = [x for x in cam_names if "cc" not in x]
        elif self.canonical_mode == 'none':
            # Remove canonical view altogether from both
            # train and valid.
            cam_names = [x for x in cam_names if "cc" not in x]
        elif self.canonical_mode == 'b0':
            cam_names = ['cc']
        else:
            raise Exception("")

        rnd_cam_name = cam_names[ np.random.randint(0, len(cam_names)) ]
        img_filename = this_filename_cc.replace("_cc", "_"+rnd_cam_name).\
            replace(".png", ".jpg")
        this_img_path = "%s/%s/images/%s" % \
            (self.root_images, subfolder, img_filename)

        img = Image.open(this_img_path).convert('RGB')
        img = self.transform(img)

        ##########
        rnd_cam_name2 = cam_names[ np.random.randint(0, len(cam_names)) ]
        img_filename2 = this_filename_cc.replace("_cc", "_"+rnd_cam_name2).\
            replace(".png", ".jpg")
        this_img_path2 = "%s/%s/images/%s" % \
            (self.root_images, subfolder, img_filename2)
        img2 = Image.open(this_img_path2).convert('RGB')
        img2 = self.transform(img2)

        #########

        this_cam = torch.FloatTensor(
            scene_from_id[rnd_cam_name]['cam_params'])

        this_cam2 = torch.FloatTensor(
            scene_from_id[rnd_cam_name2]['cam_params'])

        canonical_cam = torch.FloatTensor(
            scene_from_id['cc']['cam_params']
        )

        # Compute interesting attributes about the scene
        colors = [ elem['color'] for elem in scene_from_id['cc']['objects'] ]
        shapes = [ elem['shape'] for elem in scene_from_id['cc']['objects'] ]
        mats = [ elem['material'] for elem in scene_from_id['cc']['objects'] ]
        n_objects = len(scene_from_id['cc']['objects'])
        meta = {
            'template_filename': this_template_filename,
            'n_color_unique': len(Counter(colors)),
            'n_shape_unique': len(Counter(shapes)),
            'n_mat_unique': len(Counter(mats)),
            'n_objects': n_objects
        }

        if self.canonical_mode == 'b0':
            # Emulating vanilla clevr, so null
            # out the camera.
            this_cam = this_cam*0. + 1.

        # TODO: multiple camera views for x2_batch

        # X2_batch is None because there are no
        # alternate views per scene.
        return img, img2, this_q, this_cam, this_cam2, this_answer, meta

    def __len__(self):
        return len(self.questions)

class ClevrKiwiAutoencoderDataset(ClevrKiwiDataset):
    def __init__(self, *args, **kwargs):
        super(ClevrKiwiAutoencoderDataset, self).__init__(*args, **kwargs)
        scene_keys = list(self.id_to_scene.keys())
        len_ = len(scene_keys)
        # Make valid set 5%
        if self.mode == 'train':
            self.scene_keys = scene_keys[0:int(len_*0.95)]
        else:
            self.scene_keys = scene_keys[int(len_*0.95)::]
        print("len scene keys:", len(self.scene_keys))

    def __len__(self):
        return len(self.scene_keys)
    def __getitem__(self, index):

        # WARNING: not the same split as with VQA version


        # Randomly select a scene, and also get its subfolder
        scene_from_id = self.id_to_scene[ self.scene_keys[index] ]
        subfolder = scene_from_id['cc']['subfolder']

        fname_template = "CLEVR_train-clevr-kiwi-spatial_{scene}_{cam}.jpg"

        # Select a random camera, and create its filename
        rnd_cam_name = self.cam_names[ np.random.randint(0, len(self.cam_names)) ]
        img_filename = fname_template.format(
            scene=self.scene_keys[index],
            cam=rnd_cam_name
        )
        this_img_path = "%s/%s/images/%s" % \
            (self.root_images, subfolder, img_filename)

        img = Image.open(this_img_path).convert('RGB')
        img = self.transform(img)

        # Select another random camera, and create its filename
        rnd_cam_name2 = self.cam_names[ np.random.randint(0, len(self.cam_names)) ]
        img_filename2 = fname_template.format(
            scene=self.scene_keys[index],
            cam=rnd_cam_name2
        )
        this_img_path2 = "%s/%s/images/%s" % \
            (self.root_images, subfolder, img_filename2)
        img2 = Image.open(this_img_path2).convert('RGB')
        img2 = self.transform(img2)

        null_ = np.zeros((1,))
        this_q = null_
        this_cam = null_
        this_cam2 = null_
        this_answer = null_
        meta = {}

        return img, img2, this_q, this_cam, this_cam2, this_answer, meta

class ClevrDataset(Dataset):
    def __init__(self,
                 root_images,
                 root_meta,
                 how_many_objs=5,
                 transforms_=None,
                 mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.root_images = root_images
        self.root_meta = root_meta
        self.mode = mode

        if mode == 'train':
            f = h5py.File("%s/train_questions.h5" % root_meta, "r")
        else:
            f = h5py.File("%s/val_questions.h5" % root_meta, "r")

        self.vocab = load_vocab(os.environ["DATASET_CLEVR_META"] +
                                "/vocab.json")

        self.dat = {
            'questions': f['questions'][:],
            'image_idxs': f['image_idxs'][:],
            'answers': f['answers'][:]
        }
        f.close()

    def __getitem__(self, index):
        # Ok, grab the metadata
        this_q = torch.from_numpy(self.dat['questions'][index]).long()
        this_idx = self.dat['image_idxs'][index]
        this_answer = torch.LongTensor([self.dat['answers'][index]])
        this_img_path = "%s/%s/CLEVR_%s_%s.png" % (self.root_images,
                                                   self.mode,
                                                   self.mode,
                                                   str(this_idx).zfill(6))
        img = Image.open(this_img_path).convert('RGB')
        img = self.transform(img)

        # X2_batch is None because there are no
        # alternate views per scene.
        this_cam = torch.ones((6,)).float()

        return img, img, this_q, this_cam, this_cam, this_answer, {}

    def __len__(self):
        return len(self.dat['image_idxs'])

class ClevrRQDataset(ClevrDataset):
    def __init__(self, *args, **kwargs):
        super(ClevrRQDataset, self).__init__(*args, **kwargs)
        self.metadata = pickle.load(
            open("%s/../../clevr_rq_v3_basic.pkl" % self.root, "rb"))
        self.rnd_state = np.random.RandomState(0)

    def __getitem__(self, index):
        filepath = self.files[index]
        img = Image.open(filepath).convert('RGB')
        meta = self.metadata[os.path.basename(filepath)]
        which_idx = self.rnd_state.randint(0, len(meta))
        meta = meta[which_idx]
        #z = np.hstack((meta['mat'], meta['query']))
        z = meta['query']
        z = torch.from_numpy(z).float()
        y = meta['obj_class']
        img = self.transform(img)
        return img, z, y


class ObjectRoomsDataset(Dataset):
    def __init__(self,
                 root,
                 transforms_=None,
                 mode='train'):
        super(ObjectRoomsDataset, self).__init__()
        if mode not in ['train', 'valid', None]:
            raise Exception("`mode` must be either 'train', 'valid', or `None`!")
        dat = np.load("%s/train.npy" % root)
        N = len(dat)
        if mode == 'train':
            self.dat = dat[0:int(N*0.95)]
        else:
            self.dat = dat[int(N*0.95):]
        self.transform = transforms.Compose(transforms_)

    def __getitem__(self, index):
        img_raw = self.dat[index]
        pil_img = Image.fromarray(img_raw)
        img = self.transform(pil_img)
        return img, torch.zeros((1,1)), torch.zeros((1,1))

    def __len__(self):
        return len(self.dat)

if __name__ == '__main__':
    transforms_ = [
        transforms.Resize(224),
        transforms.ToTensor()
    ]
    ds = ClevrDataset(root_images="/clevr/CLEVR_v1.0/images/",
                      root_meta="/clevr_preprocessed/",
                      transforms_=transforms_)

    from torch.utils.data import  DataLoader
    loader = DataLoader(ds, num_workers=0, batch_size=16)

    for x_batch,q_batch,y_batch in loader:
        break

    import pdb
    pdb.set_trace()
