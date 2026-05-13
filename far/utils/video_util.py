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

import os
import random
import tempfile
from io import BytesIO
from typing import Optional

import imageio
import numpy as np
from PIL import Image


class VideoLoader_Imageio_Backend:
    def __init__(
        self,
        fp: str | BytesIO,
    ):
        if isinstance(fp, str):
            self.video_reader = imageio.get_reader(fp)
        elif isinstance(fp, BytesIO):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(fp.read())
                temp_video.flush()
                self.temp_video_name = temp_video.name
                self.video_reader = imageio.get_reader(self.temp_video_name)
        else:
            raise ValueError(f'Type {type(fp)} is not supported')

    def close(self):
        if hasattr(self, 'temp_video_name'):
            if os.path.exists(self.temp_video_name):
                os.remove(self.temp_video_name)

    def get_frame_count(self) -> int:
        """
        Returns the total number of frames in the video.

        Returns:
            int: The number of frames.
        """
        return self.video_reader.count_frames()

    def get_fps(self) -> Optional[float]:
        """
        Returns the frames per second (FPS) of the video.

        Returns:
            Optional[float]: The FPS of the video, or None if the data is not available.
        """
        meta_data = self.video_reader.get_meta_data()
        return meta_data.get('fps', 30)

    def get_frames(self, num_frames: int, target_fps=None) -> list[Image.Image]:
        original_fps = self.get_fps()
        total_frames = self.get_frame_count()

        if target_fps is None:
            target_fps = original_fps

        frame_idxs = select_frame_indices(
            total_frames,
            original_fps,
            target_fps=target_fps
        )

        frame_idxs = frame_idxs[:num_frames]
        frames = [self.video_reader.get_data(idx) for idx in frame_idxs]  # (480, 854, 3) uint8
        return np.stack(frames)  # print(len(frames), np.stack(frames).shape) # (81, 480, 854, 3)


class VideoLoader:
    def __init__(
        self,
        fp: str | BytesIO,
        name: Optional[str] = None,
        crop_range: Optional[tuple[float, float, float, float]] = None,
    ):
        import decord
        decord.bridge.set_bridge('torch')

        if isinstance(fp, str):
            self.video_reader = decord.VideoReader(fp)
            name = name or fp
        elif isinstance(fp, BytesIO):
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_video:
                temp_video.write(fp.read())
                temp_video.flush()
                temp_video_name = temp_video.name
                self.video_reader = decord.VideoReader(temp_video_name)
                name = name or temp_video_name
        else:
            raise ValueError(f'Type {type(fp)} is not supported')
        self.name = name

    def get_frame_count(self):
        return len(self.video_reader)

    def random_sample_frames(self, total_frames, num_frames, interval=1, random_start=False):
        if total_frames < (num_frames - 1) * interval + 1:
            interval = 1
        max_start = total_frames - ((num_frames - 1) * interval + 1)
        if random_start:
            start = random.randint(0, max_start)
        else:
            start = 0
        frame_ids = [start + i * interval for i in range(num_frames)]
        return frame_ids

    def get_contiguous_frames(self, num_frames: int) -> list[Image.Image]:
        frame_idxs = self.random_sample_frames(self.get_frame_count(), num_frames)
        frames = self.video_reader.get_batch(frame_idxs)
        return frames


def select_frame_indices(total_frames, original_fps, target_fps):

    accumulator_step = target_fps / original_fps
    accumulator = 0.0

    frame_idxs = [0]

    for i in range(1, total_frames):
        accumulator += accumulator_step

        if accumulator >= 1.0:
            frame_idxs.append(i)
            accumulator -= 1.0

    return frame_idxs
