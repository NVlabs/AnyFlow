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

import getpass
import io
import json
import mmap
import os
import tarfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize
from tqdm import tqdm

from far.utils.video_util import VideoLoader_Imageio_Backend


def generate_tar(data_format: Dict[str, str], data_list: List[Tuple[str, Dict[str, Any]]], tar_path: str) -> None:
    """
    data_format: `key` is suffix like ".png", `value` should be str from ["file_path", "data"], "file_path" means `data[key]` is a file path, "data" means `data[key]` is raw data.  # noqa: E501
    data_list: List of (prefix, data). For each data, `key` should be in `data_format`, `data[key]` should be either a file path or raw data.
    """
    tar_path = os.path.abspath(tar_path)
    tar = tarfile.open(tar_path, 'w')
    for prefix, data in tqdm(data_list, desc=f'generate {tar_path}', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        for key in data_format:
            if data_format[key] == 'file_path':
                tar.add(data[key], arcname=prefix + key)
            elif data_format[key] == 'data':
                if key == '.json':
                    fileobj = io.BytesIO(json.dumps(data[key]).encode('utf-8'))
                elif key == '.jpg':
                    if isinstance(data[key], Image.Image):
                        fileobj = io.BytesIO()
                        data[key].save(fileobj, format='JPEG')
                        fileobj.seek(0)
                    elif isinstance(data[key], io.BytesIO):
                        fileobj = data[key]
                    else:
                        raise ValueError(f'type {type(data[key])} is not supported for .jpg')
                elif key == '.npy':
                    fileobj = io.BytesIO()
                    np.save(fileobj, data[key])
                    fileobj.seek(0)
                elif key == '.npz':
                    fileobj = io.BytesIO()
                    np.savez(fileobj, **data[key])
                    fileobj.seek(0)
                elif key == '.pth':
                    fileobj = io.BytesIO()
                    torch.save(data[key], fileobj)
                    fileobj.seek(0)
                else:
                    raise ValueError(f'{key} is not supported as raw data')
                tar_info = tarfile.TarInfo(name=prefix + key)
                tar_info.size = fileobj.getbuffer().nbytes
                tar_info.mtime = int(time.time())  # avoids large header size
                tar_info.uname = getpass.getuser()
                tar_info.gname = 'dip'
                tar.addfile(tar_info, fileobj)
            else:
                raise ValueError(f'data format for {key} {data_format[key]} is not supported')

    tar.close()


class SingleTarDataset(Dataset):
    def __init__(self, tar_path: str, height: int, width: int, num_frames: int, target_fps: int):
        with tarfile.open(tar_path, 'r') as tar_file:
            self.tar_stream = open(tar_path, 'rb')
            self.sample_prefix_list = []
            self.sample_meta_list = []
            last_prefix = ''
            self.mmapped_file = mmap.mmap(
                self.tar_stream.fileno(), 0, access=mmap.ACCESS_READ
            )  # mmap is necessary since tarfile doesn't work with multiprocessing
            for tarinfo in tar_file:
                prefix, ext = os.path.splitext(tarinfo.name)
                if prefix != last_prefix:
                    self.sample_meta_list.append({})
                    self.sample_prefix_list.append(prefix)
                self.sample_meta_list[-1][ext] = (tarinfo.name, tarinfo.size, tar_file.fileobj.tell())
                last_prefix = prefix

        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.target_fps = target_fps

    def __len__(self):
        return len(self.sample_meta_list)

    def __getitem__(self, index: int):
        sample = {'__key__': self.sample_prefix_list[index]}
        try:
            for ext in self.sample_meta_list[index]:
                name, size, offset = self.sample_meta_list[index][ext]
                stream = io.BytesIO(self.mmapped_file[offset: offset + size])
                if ext == '.json':
                    sample[ext] = json.load(stream)
                elif ext in ['.jpg', '.jpeg', '.png', '.ppm', '.pgm', '.pbm', '.pnm', '.webp', '.bmp', '.tiff']:
                    sample[ext] = Image.open(stream)
                elif ext == '.npy':
                    sample[ext] = np.load(stream)
                elif ext == '.mp4':
                    sample[ext] = VideoLoader_Imageio_Backend(stream)
                    video = sample[ext].get_frames(self.num_frames, self.target_fps)

                    sample[ext].close()
                    del sample[ext]

                    video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
                    t, _, h, w = video.shape

                    source_aspect_ratio = 1.0 * w / h
                    target_aspect_ratio = 1.0 * self.width / self.height

                    if t != self.num_frames:
                        print(f"{sample['__key__']} only have {t} frames, less than required {self.num_frames} frames, skip")
                        raise NotImplementedError
                    elif not ((source_aspect_ratio <= target_aspect_ratio + 0.1) and (source_aspect_ratio >= target_aspect_ratio - 0.1)):
                        print(f"{sample['__key__']} is {source_aspect_ratio:2f} aspect ratio, but require {target_aspect_ratio:2f} aspect ratio, skip")
                        raise NotImplementedError
                    else:
                        transform = Compose([
                            Resize((self.height, self.width)),
                            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                        ])
                        sample['video'] = transform(video)
                else:
                    raise ValueError(f'Unsupported ext: {ext}')
        except:
            print(f"Error reading {sample['__key__']}, skip")
            sample = None
        return sample

    def __del__(self):
        self.mmapped_file.close()
        self.tar_stream.close()
