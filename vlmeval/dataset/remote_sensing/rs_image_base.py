import multiprocessing as mp
import os.path as osp
import warnings
from functools import partial

import numpy as np

from ...smp import d2df, dump, listinstr, load, toliststr
from ..image_base import ImageBaseDataset


class RSImageBaseDataset(ImageBaseDataset):
    """Compared with the original implementation, the remote sensing image
    dataset store image_path instead of encoded base64 image.
    """

    def __init__(self, image_root=None, **kwargs):
        super().__init__(**kwargs)
        self.image_root = image_root

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            assert self.image_root is not None
            # create full image path
            tgt_path = [
                osp.join(self.image_root, p)
                for p in toliststr(line['image_path'])
            ]
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs
