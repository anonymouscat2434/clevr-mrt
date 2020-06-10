import os
from torch.utils.data import (DataLoader,
                              Subset)
from iterators.datasets import (ClevrDataset,
                                ClevrKiwiDataset,
                                ClevrKiwiAutoencoderDataset,
                                BlankDataset)
from torchvision.transforms import transforms

def load_dataset(name,
                 img_size,
                 imagenet_scaling=False,
                 train=True):

    if imagenet_scaling:
        norm_ = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.224))
    else:
        norm_ = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))

    if name == 'clevr':
        # https://github.com/ethanjperez/film/blob/master/scripts/extract_features.py#L58
        train_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm_
        ]
        if train:
            ds_train = ClevrDataset(root_images=os.environ['DATASET_CLEVR_IMAGES'],
                                    root_meta=os.environ['DATASET_CLEVR_META'],
                                    transforms_=train_transforms,
                                    mode='train')
            ds_valid = ClevrDataset(root_images=os.environ['DATASET_CLEVR_IMAGES'],
                                    root_meta=os.environ['DATASET_CLEVR_META'],
                                    transforms_=train_transforms,
                                    mode='val')
        else:
            raise Exception("test set not yet added for clevr")
    elif name in ['clevr_kiwi', 'clevr_kiwi_cc', 'clevr_kiwi_nocc']:
        # https://github.com/ethanjperez/film/blob/master/scripts/extract_features.py#L58
        train_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm_
        ]

        print("Transform:")
        print("  " + str(train_transforms))

        canonical_mode = True if '_cc' in name else False

        if name == 'clevr_kiwi':
            # Only have it in train, don't
            # use cc in valid.
            canonical_mode = 'train_only'
        elif name == 'clevr_kiwi_cc':
            # Canonical view ONLY in train and valid
            # ('baseline 0', basically emulate
            # vanilla clevr)
            canonical_mode = 'b0'
        elif name == 'clevr_kiwi_nocc':
            # No canonical views used in train nor valid.
            # Should be the hardest option.
            canonical_mode = 'none'
        else:
            raise Exception("")

        import pickle

        if train:
            ds_train = ClevrKiwiDataset(root_images=os.environ['DATASET_CLEVR_KIWI'],
                                        root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                                        transforms_=train_transforms,
                                        canonical_mode=canonical_mode,
                                        mode='train')
            ds_valid = ClevrKiwiDataset(root_images=os.environ['DATASET_CLEVR_KIWI'],
                                        root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                                        transforms_=train_transforms,
                                        canonical_mode=canonical_mode,
                                        mode='val')
        else:
            ds_test = ClevrKiwiDataset(root_images=os.environ['DATASET_CLEVR_KIWI_TEST'],
                                       root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                                       transforms_=train_transforms,
                                       canonical_mode=canonical_mode,
                                       mode='test')
    elif name == 'clevr_kiwi_ae':

        train_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm_
        ]

        import pickle
        root_meta = os.environ['DATASET_CLEVR_KIWI_META']
        cache_file = "%s/cache.pkl" % root_meta
        if os.path.exists(cache_file):
            # Load id_to_scene from cache
            with open("%s/cache.pkl" % (root_meta), "rb") as f_read:
                cache_pkl = pickle.load(f_read)
        else:
            cache_pkl = None

        if train:
            ds_train = ClevrKiwiAutoencoderDataset(root_images=os.environ['DATASET_CLEVR_KIWI'],
                                                   root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                                                   transforms_=train_transforms,
                                                   canonical_mode='train_only',
                                                   cache_pkl=cache_pkl,
                                                   mode='train')
            ds_valid = ClevrKiwiAutoencoderDataset(root_images=os.environ['DATASET_CLEVR_KIWI'],
                                                   root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                                                   transforms_=train_transforms,
                                                   canonical_mode='train_only',
                                                   cache_pkl=cache_pkl,
                                                   mode='val')
        else:
            ds_test = ClevrKiwiAutoencoderDataset(root_images=os.environ['DATASET_CLEVR_KIWI_TEST'],
                                                  root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                                                  transforms_=train_transforms,
                                                  canonical_mode='train_only',
                                                  cache_pkl=cache_pkl,
                                                  mode='test')
    
    elif name == 'blank':

        from iterators.utils import load_vocab
        vocab = load_vocab(os.environ["DATASET_CLEVR_KIWI_META"] +
                           "/vocab.json")
        if train:
            ds_train = BlankDataset()
            ds_valid = BlankDataset()
            ds_train.vocab = vocab
            ds_valid.vocab = vocab
        else:
            ds_test = BlankDataset()
            ds_test.vocab = vocab

    else:
        raise Exception("Specified dataset %s is not valid" % name)

    if train:
        return ds_train, ds_valid
    else:
        return ds_test
