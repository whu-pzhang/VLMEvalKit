import multiprocessing as mp
import warnings
from functools import partial

import numpy as np

from ...smp import d2df, dump, listinstr, load
from ..image_base import ImageBaseDataset
from .rs_image_base import RSImageBaseDataset


class SceneClassificationDataset(RSImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'AID':
        'https://modelscope.cn/datasets/pzhang199/RSBench/resolve/master/aid.tsv'
    }
    DATASET_MD5 = {'AID': 'ac530be0e37987aabf14a5922f311578'}

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1][
            'value'] += '\nAnswer the question using a single word or phrase.'
        return msgs

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        from ..utils.vqa_eval import hit_calculate, process_line

        data = load(eval_file)
        dataset = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        lt = len(data)
        pool = mp.Pool(4)
        lines = [data.iloc[i] for i in range(lt)]

        res = pool.map(partial(process_line, method='accuracy'), lines)

        hit = hit_calculate(res, dataset)
        ret = dict()
        if 'split' in data:
            splits = set(data['split'])
            for sp in splits:
                sub = [r for l, r in zip(lines, res) if l['split'] == sp]
                # [np.mean(x['match']) >= full_score_weight for x in sub]
                hit = hit_calculate(sub, dataset)
                ret[sp] = np.mean(hit) * 100
            sub = [r for l, r in zip(lines, res)]
            hit = hit_calculate(sub, dataset)
            ret['Overall'] = np.mean(hit) * 100
        else:
            ret['Overall'] = np.mean(hit) * 100
            if 'category' in data:
                cates = list(set(data['category']))
                cates.sort()
                for c in cates:
                    sub = [r for l, r in zip(lines, res) if l['category'] == c]
                    # [np.mean(x['match']) >= full_score_weight for x in sub]
                    hit = hit_calculate(sub, dataset)
                    ret[c] = np.mean(hit) * 100
        ret = d2df(ret)
        ret.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret
