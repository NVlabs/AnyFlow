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

os.environ['VBENCH_CACHE_DIR'] = 'experiments/pretrained_models/vbench'


class VBenchI2VEvaluator:
    METRICS_NORMALIZATION_RANGES = {
        'subject_consistency': [0.1462, 1.0],
        'motion_smoothness': [0.706, 0.9975],
        'temporal_flickering': [0.6293, 1.0],
        'background_consistency': [0.2615, 1.0],
        'scene': [0.0, 0.8222],
        'appearance_style': [0.0009, 0.2855],
        'temporal_style': [0.0, 0.364],
        'overall_consistency': [0.0, 0.364],
        'i2v_subject': [0.1462, 1.0],
        'i2v_background': [0.2615, 1.0]
    }

    DIMENSIONS = [
        'camera_motion', 'i2v_subject', 'i2v_background',
        'subject_consistency', 'motion_smoothness', 'background_consistency',
        'dynamic_degree', 'aesthetic_quality', 'imaging_quality'
    ]

    def __init__(
        self,
        save_root_dir,
        resolution,
        device
    ):
        from vbench2_beta_i2v import VBenchI2V  # noqa: E402
        self.evaluator = VBenchI2V(device, 'assets/data/meta/vbench/vbench2_i2v_full_info.json', f'{save_root_dir}/vbench_info')
        self.resolution = resolution
        self.vbench_info_dir = f'{save_root_dir}/vbench_info'

    def evaluate(self, result_dir):

        eval_info_dict = {}
        for metric_dimension in self.DIMENSIONS:
            vbench_info_path = os.path.join(self.vbench_info_dir, f'{metric_dimension}_eval_results.json')
            if os.path.exists(vbench_info_path):
                with open(vbench_info_path, 'r') as fr:
                    metric_dict = json.load(fr)
                    eval_info_dict[metric_dimension] = metric_dict[metric_dimension][0]
            else:
                eval_info_dict.update(self.evaluator.evaluate(
                    videos_path=f'{result_dir}/samples',
                    name=f'{metric_dimension}',
                    dimension_list=[metric_dimension],
                    resolution=self.resolution,
                    local=True,
                ))

        eval_info_dict['quality_score'] = (
            self._norm(eval_info_dict['subject_consistency'], 'subject_consistency') +  # noqa: W504
            self._norm(eval_info_dict['background_consistency'], 'background_consistency') +  # noqa: W504
            self._norm(eval_info_dict['motion_smoothness'], 'motion_smoothness') +  # noqa: W504
            self._norm(eval_info_dict['dynamic_degree'], 'dynamic_degree') * 0.5 +  # noqa: W504
            self._norm(eval_info_dict['aesthetic_quality'], 'aesthetic_quality') +  # noqa: W504
            self._norm(eval_info_dict['imaging_quality'], 'imaging_quality')  # noqa: W504
        ) / 5.5  # noqa: W504

        eval_info_dict['i2v_score'] = (
            self._norm(eval_info_dict['i2v_subject'], 'i2v_subject') +  # noqa: W504
            self._norm(eval_info_dict['i2v_background'], 'i2v_background') +  # noqa: W504
            self._norm(eval_info_dict['camera_motion'], 'camera_motion') * 0.1
        ) / 2.1

        eval_info_dict['overall_score'] = 0.5 * eval_info_dict['i2v_score'] + 0.5 * eval_info_dict['quality_score']
        return eval_info_dict

    def _norm(self, metric, key):
        range = self.METRICS_NORMALIZATION_RANGES[key] if key in self.METRICS_NORMALIZATION_RANGES else [0.0, 1.0]
        metric = max(metric, range[0])
        metric = min(metric, range[1])
        metric = (metric - range[0]) / (range[1] - range[0])
        return metric
