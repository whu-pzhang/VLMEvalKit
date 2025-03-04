"""
Convert DIOR-RSVG to TSV format.
"""
import csv
import re
from collections import OrderedDict
from pathlib import Path
from typing import Union

from tqdm import tqdm

from vlmeval.smp import load
from vlmeval.smp.vlm import encode_image_file_to_base64


def convert_diorvg_to_tsv(json_file: Union[str, Path],
                          image_dir: Union[str, Path],
                          target_file: Union[str, Path],
                          with_image: bool = False):

    image_root = Path(image_dir)
    ann_data = load(json_file)

    def sample2row(sample):
        image_path = image_root / sample['image']
        question = re.sub(r'</?ref[^>]*>', '', sample['prompt'])
        answer = sample['bbox']
        image_size = sample['size']
        image_path_rel = sample['image']
        if with_image:
            image = encode_image_file_to_base64(image_path)
            return dict(question=question,
                        answer=answer,
                        image_path=image_path_rel,
                        image=image)
        else:
            return dict(question=question,
                        answer=answer,
                        image_path=image_path_rel)

    fp = open(target_file, 'a', encoding='utf8')
    header = ['index', 'question', 'answer', 'category', 'image_path']
    if with_image:
        header.append('image')

    tsv_writer = csv.writer(fp, delimiter='\t')
    tsv_writer.writerow(header)

    for idx, sample in enumerate(tqdm(ann_data)):
        line = sample2row(sample)
        record = [
            idx, line['question'], line['answer'], 'REF', line['image_path']
        ]
        if with_image:
            record.append(line['image'])
        tsv_writer.writerow(record)
    fp.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Convert AID dataset to TSV format')
    parser.add_argument(
        'json_file',
        type=str,
        default=
        '/data2/vlm_data/OpenGVLab/InternVL-Domain-Adaptation-Data/val/dior_rsvg_test.json'
    )
    parser.add_argument('image_root',
                        type=str,
                        default='/data2/vlm_data/RS_Images/')
    parser.add_argument('target_file',
                        type=str,
                        default='/data2/vlm_data/dior_rsvg_test.tsv')
    parser.add_argument('--with-image',
                        action='store_true',
                        default=False,
                        help='encode image into tsv')
    args = parser.parse_args()
    convert_diorvg_to_tsv(args.json_file, args.image_root, args.target_file,
                          args.with_image)
