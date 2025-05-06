import concurrent
import glob
import json
import multiprocessing
import os
from collections import defaultdict

import cv2
import fiftyone as fo
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image

from .pp_utils.datatypes import CameraParams, ObjectDetection
from .pp_utils.utils import _read_exr
from .pp_utils.utils_3d import proj_3dbbox_to_image_coords


class BasicDSParser:
    def __init__(self, data_dir, config=None) -> None:
        self.data_dir = data_dir
        self.config = config
        self.session_hash = None
        self.cameras = None
        self._groups = defaultdict(fo.Group)
        self.grouped_dataset = None

    def setup(self):
        g = os.walk(f"{self.data_dir}/frames")
        self.session_hash = next(g)[1]
        self.cameras = next(g)[1]
        self.grouped_dataset = len(self.cameras) > 1
        assert len(self.session_hash) == 1, "Multiple session hashes found"
        assert len(self.cameras) >= 1, "At least one camera expected"

        self.session_hash = self.session_hash[0]

    def _get_camera_path(self, camera):
        return os.path.join(self.data_dir, "frames", self.session_hash, camera)

    def parse_dataset(self):
        samples = []
        for camera in self.cameras:
            print(f"{camera=}")
            camera_path = self._get_camera_path(camera)
            images = glob.glob(os.path.join(camera_path, "ldr_color/*"))
            if not len(images) >= 1:
                print(f"{camera} folder doesn't contain any RGB data")
                continue

            os.makedirs(os.path.join(camera_path, "semantic_segmentation_fo"), exist_ok=True)
            os.makedirs(os.path.join(camera_path, "depth_fo"), exist_ok=True)

            lock = multiprocessing.Lock()

            with tqdm.tqdm(len(images)) as pbar:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [executor.submit(self.parse_frame, camera_path, image) for image in images]

                    for _idx, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            result, frame_num = future.result()
                        except Exception as e:
                            print(f"Error parsing frame: {e}")
                            continue

                        sample = fo.Sample.from_dict(result)
                        result["camera"] = camera
                        if self.grouped_dataset:
                            with lock:
                                sample.set_field("group", self._groups[frame_num].element(camera))
                                samples.append(sample)
                                pbar.update(1)
                        else:
                            samples.append(sample)
                            pbar.update(1)

        return samples

    def parse_2d_bbox(self, camera_path, filename):
        bbox_data = np.load(os.path.join(camera_path, f"bounding_box_2d_tight/{filename}.npy"))

        with open(os.path.join(camera_path, f"bounding_box_2d_tight/labels_{filename}.json")) as file:
            labels_data = json.load(file)

        with open(os.path.join(camera_path, f"camera_params/{filename}.json")) as file:
            camera_params = json.load(file)

        camera_params = CameraParams(camera_params)

        im_height = camera_params.cameraFisheyeNominalHeight
        im_width = camera_params.cameraFisheyeNominalWidth

        if im_width == 0 or im_height == 0:
            print(f"Skipping 2D bbox for {filename} due to zero image size")
            return fo.Detections(detections=[])

        num_detections = bbox_data["semanticId"].shape[0]

        bbox_data_norm = np.array(
            [
                bbox_data["x_min"].astype(int) / im_width,
                bbox_data["y_min"].astype(int) / im_height,
                (bbox_data["x_max"].astype(int) - bbox_data["x_min"].astype(int)) / im_width,
                (bbox_data["y_max"].astype(int) - bbox_data["y_min"].astype(int)) / im_height,
            ]
        )

        detections = []
        for i in range(num_detections):
            label = labels_data[f"{bbox_data['semanticId'][i]}"]["class"]
            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=bbox_data_norm[:, i].tolist(),
                )
            )

        return fo.Detections(detections=detections)

    def parse_semantic_segmentation(self, camera_path, filename):
        mask_targets = {}
        sem_seg = plt.imread(os.path.join(camera_path, "semantic_segmentation", f"{filename}.png"))

        with open(os.path.join(camera_path, "semantic_segmentation", f"mapping_{filename}.json")) as file:
            mapping = json.load(file)

        for k in mapping.keys():
            if k not in mask_targets:
                mask_targets[int(k)] = mapping[k]["class"]

        seg_path = os.path.join(camera_path, "semantic_segmentation_fo", f"{filename}.png")
        cv2.imwrite(seg_path, sem_seg[:, :, 0] * 255)

        return seg_path, mask_targets

    def parse_depth(self, camera_path, filename):
        NEAR = 0.01
        FAR = 100

        depth_data = _read_exr(os.path.join(camera_path, "distance_to_camera", f"{filename}.exr"))
        depth_data = np.clip(depth_data, NEAR, FAR)
        depth_data = (np.log(depth_data) - np.log(NEAR)) / (np.log(FAR) - np.log(NEAR))
        depth_data_uint8 = (depth_data * 255).astype(np.uint8)

        img = Image.fromarray(depth_data_uint8)
        map_path = os.path.join(camera_path, "depth_fo", f"{filename}.png")
        img.save(map_path)

        return map_path

    def parse_frame(self, camera_path, filename):
        img_path = os.path.abspath(filename)
        filename = os.path.splitext(os.path.basename(filename))[0]

        sample = {"filepath": img_path}

        if os.path.exists(os.path.join(camera_path, "bounding_box_2d_tight")):
            sample["bbox2d"] = self.parse_2d_bbox(camera_path, filename)

        # ✅ 3D bounding box parsing intentionally skipped
        # if os.path.exists(os.path.join(camera_path, "bounding_box_3d")):
        #     sample["bbox3d"] = self.parse_3d_bbox(camera_path, filename)

        if os.path.exists(os.path.join(camera_path, "semantic_segmentation")):
            mask_path, mask_targets = self.parse_semantic_segmentation(camera_path, filename)
            sample["segmentations"] = fo.Segmentation(mask_path=mask_path)

        if os.path.exists(os.path.join(camera_path, "distance_to_camera")):
            map_path = self.parse_depth(camera_path, filename)
            sample["depth"] = fo.Heatmap(map_path=map_path)

        return sample, filename
