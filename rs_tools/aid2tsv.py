import csv
from pathlib import Path
from typing import Union

from tqdm import tqdm

from vlmeval.smp.vlm import encode_image_file_to_base64


def convert_aid_to_tsv(data_root: Union[str, Path],
                       target_file: Union[str, Path],
                       with_image: bool = False):
    prompt = 'Classify the given image in one of the following classes: '
    # Get class names from the folder names
    classes = sorted([p.name for p in Path(data_root).iterdir()])

    if Path(target_file).exists():
        print(f'Target file {target_file} already exists. Skip conversion.')
        return

    fp = open(target_file, 'a', encoding='utf8')
    header = ['index', 'question', 'answer', 'category', 'image_path']
    if with_image:
        header.append('image')

    tsv_writer = csv.writer(fp, delimiter='\t')
    tsv_writer.writerow(header)
    image_list = sorted(Path(data_root).rglob('*.jpg'))
    for idx, image_path in enumerate(tqdm(image_list)):
        question = prompt + ', '.join(classes)
        answer = image_path.parent.name
        category = answer  # using class name as category
        image_path_rel = image_path.relative_to(data_root).as_posix()
        record = [idx, question, answer, category, image_path_rel]
        if with_image:
            image = encode_image_file_to_base64(image_path)
            record.append(image)
        tsv_writer.writerow(record)

    fp.close()


def main():
    import argparse

    parser = argparse.ArgumentParser('Convert AID dataset to TSV format')
    parser.add_argument('data_root',
                        type=str,
                        default='/data2/vlm_data/RS_Images/AID')
    parser.add_argument('target_file', type=str, default='./aid.tsv')
    parser.add_argument('--with-image',
                        action='store_true',
                        default=False,
                        help='encode image into tsv')
    args = parser.parse_args()

    convert_aid_to_tsv(args.data_root, args.target_file, args.with_image)


if __name__ == '__main__':
    main()
