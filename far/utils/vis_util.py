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

import cv2
import numpy as np


def draw_rectangle(video, context_length):
    for frame_idx, frame in enumerate(video):
        if frame_idx < context_length:
            frame_uint8 = np.ascontiguousarray((frame * 255).astype(np.uint8))
            frame_with_rect_uint8 = cv2.rectangle(frame_uint8, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (255, 0, 0), 5)
            video[frame_idx] = frame_with_rect_uint8.astype(np.float32) / 255.0
        else:
            video[frame_idx] = frame
    return video
