import concurrent
import glob
import json
import multiprocessing
import os
from collections import defaultdict
from string import Template

import fiftyone as fo
import numpy as np
import tqdm

HEADER = Template(
    """
# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH $num_points
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS $num_points
DATA ascii
"""
)


class PointCloudParser:
    def __init__(self, data_dir, config=None) -> None:
        self.data_dir = data_dir
        self.config = config
        self.session_hash = None
        self.cameras = None
        self.lidar_pcd_path = None
        self._groups = defaultdict(fo.Group)
        self.grouped_dataset = None

        # Store frame ids
        self._frame_ids = set()

    def setup(self):
        g = os.walk(f"{self.data_dir}/frames")
        self.cameras = next(g)[1]
        self.grouped_dataset = len(self.cameras) > 1
        assert len(self.cameras) >= 1, "At least one camera expected"

        # Create directories to store pcd files
        self.lidar_pcd_path = f"{self.data_dir}/lidar_pcd"
        os.makedirs(self.lidar_pcd_path, exist_ok=True)

    def _get_camera_path(self, camera):
        return os.path.join(self.data_dir, "frames", camera)

    def _get_lidar_path(self):
        return os.path.join(self.data_dir, "lidar")

    def parse_dataset(self):
        samples = []
        # Parse camera data
        for camera in self.cameras:
            print(f"{camera=}")
            camera_path = self._get_camera_path(camera)
            images = glob.glob(os.path.join(camera_path, "ldr_color/*"))

            if not len(images) >= 1:
                print(f"{camera} folder doesn't contain any RGB data")
                continue

            lock = multiprocessing.Lock()

            with tqdm.tqdm(len(images)) as pbar:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [executor.submit(self.parse_frame, camera_path, image) for image in images]

                    for _idx, future in enumerate(concurrent.futures.as_completed(futures)):
                        result, frame_num = future.result()
                        sample = fo.Sample.from_dict(result)
                        self._frame_ids.add(frame_num)
                        if self.grouped_dataset:
                            with lock:
                                sample.set_field("group", self._groups[frame_num].element(camera))
                                samples.append(sample)
                                pbar.update(1)
                        else:
                            samples.append(sample)
                            pbar.update(1)

        # Parse lidar data
        with tqdm.tqdm(len(self._frame_ids)) as pbar:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.parse_lidar, frame_num) for frame_num in self._frame_ids]

                for _idx, future in enumerate(concurrent.futures.as_completed(futures)):
                    result, frame_num = future.result()
                    sample = fo.Sample.from_dict(result)

                    with lock:
                        sample.set_field("group", self._groups[frame_num].element("pcd"))
                        samples.append(sample)
                        pbar.update(1)

        return samples

    def convert_npy_pcd(self, lidar_num):
        npy_path = os.path.join(self._get_lidar_path(), f"{lidar_num}.npy")
        data = np.load(npy_path)
        num_points = data.shape[0]

        out_path = os.path.join(self.lidar_pcd_path, f"{lidar_num}.pcd")
        np.savetxt(
            out_path,
            np.column_stack((data["x"], data["y"], data["z"])),
            delimiter=" ",
            header=HEADER.substitute(num_points=num_points),
            comments="",
            fmt="%s",
        )

        return out_path

    def read_pointcloud_bbox(self, lidar_num):
        bbox_path = os.path.join(self._get_camera_path("lidar_front"), "bounding_box_3d_360", f"{lidar_num}.json")

        with open(bbox_path, "r") as file:
            data = json.load(file)

        detections = []

        for bbox in data:
            detections.append(
                fo.Detection(
                    label=bbox["label"],
                    location=bbox["shape3d"]["cuboid3d"]["center"],
                    dimensions=bbox["shape3d"]["cuboid3d"]["dimensions"],
                    rotation=bbox["shape3d"]["cuboid3d"]["orientation"],
                    prim=bbox["prim"],
                )
            )

        return fo.Detections(detections=detections)

    def parse_frame(self, camera_path, filename):
        img_path = os.path.abspath(filename)
        filename = os.path.splitext(os.path.basename(filename))[0]

        sample = {"filepath": img_path}

        return sample, filename

    def parse_lidar(self, frame_num):
        # Divide the RGB frame number by 3 to get the corresponding lidar file
        lidar_num = int(frame_num) // 3

        pcd_path = self.convert_npy_pcd(lidar_num)
        detections = self.read_pointcloud_bbox(lidar_num)

        sample = {"filepath": pcd_path, "detections": detections}

        return sample, frame_num
