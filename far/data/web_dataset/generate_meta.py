# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tarfile
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional

from omegaconf import MISSING

from .wids import WebDataset


def generate_and_load_tar_meta(tar_path: str, cache_dir: str, overwrite: bool = False) -> dict:
    tar_meta_path = os.path.join(
        os.path.expanduser(cache_dir),
        tar_path.replace('/', '--') + '.json',
    )

    if not os.path.exists(tar_meta_path) or overwrite:
        print(f'Generating meta: {tar_meta_path}')
        try:
            tar = tarfile.open(tar_path)
            uuids = set([os.path.splitext(_)[0] for _ in tar.getnames()])
            if '.' in uuids:
                uuids.remove('.')  # for sam
        except tarfile.ReadError as e:
            print(f'Skipping {tar_path}')
            print(e)
            return None
        nsamples = len(uuids)

        tar_meta = {
            'url': tar_path,
            'nsamples': nsamples,
            'filesize': os.path.getsize(tar_path),
        }
        os.makedirs(os.path.dirname(tar_meta_path), exist_ok=True)
        json.dump(tar_meta, open(tar_meta_path, 'w'), indent=4)

    print(f'Loading abs meta: {tar_meta_path}')
    tar_meta = json.load(open(tar_meta_path, 'r'))
    return tar_meta


def generate_meta(
    data_dir: str, cache_dir: str = '~/.cache/web_dataset_meta', save_path: Optional[str] = None, processes: int = 10
) -> None:
    data_dir = os.path.abspath(data_dir)
    cache_dir = os.path.expanduser(cache_dir)

    tar_path_list = []
    for root, _, file_names in os.walk(data_dir):
        for file_name in file_names:
            if not file_name.endswith('.tar'):
                continue
            file_path = os.path.join(root, file_name)
            tar_path_list.append(file_path)
    tar_path_list = sorted(tar_path_list)

    assert len(tar_path_list) > 0, f'no tar was found in the repository {data_dir} !'
    print(f'generating meta for total {len(tar_path_list)} files.')

    with Pool(processes=processes) as pool:
        args_list = [(tar_path, cache_dir) for tar_path in tar_path_list]
        tar_meta_list = pool.starmap(generate_and_load_tar_meta, args_list)

    if save_path is None:
        save_path = os.path.join(data_dir, 'wids-meta.json')

    meta = {
        'wids_version': 1,
        'shardlist': sorted(tar_meta_list, key=lambda x: x['url']),
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(meta, open(save_path, 'w'), indent=4)


@dataclass
class GenerateMetaConfig:
    data_dir: str = MISSING
    save_path: Optional[str] = None
    test_load_data: bool = False


def main():
    from omegaconf import OmegaConf

    cfg: GenerateMetaConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(GenerateMetaConfig), OmegaConf.from_cli())
    )

    generate_meta(cfg.data_dir, save_path=cfg.save_path)

    if cfg.test_load_data:
        dataset = WebDataset(cfg.data_dir, cfg.save_path)
        print(f'dataset size: {len(dataset)}')
        print(dataset[0])

        # data_loader = torch.utils.data.DataLoader(
        #     dataset,
        #     shuffle=False,
        #     batch_size=8,
        #     num_workers=8,
        # )
        # for idx, data in tqdm(enumerate(data_loader)):
        #     pass


if __name__ == '__main__':
    main()

"""
python -m far.data.web_dataset.generate_meta data_dir=datasets/openvid-hd-wan-latents-81frames-tarsier2_recaption save_path=assets/data/meta/openvid/openvid-hd.json"""
