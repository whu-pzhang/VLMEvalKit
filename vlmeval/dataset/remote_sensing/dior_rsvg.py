import warnings

from ..image_base import ImageBaseDataset


class DiorRSVGDataset(ImageBaseDataset):
    TYPE = 'REF'

    DATASET_URL = {}
    DATASET_MD5 = {}

    def build_prompt(self, line):
        return super().build_prompt(line)

    def evaluate(self, eval_file, **judge_kwargs):
        return super().evaluate(eval_file, **judge_kwargs)
