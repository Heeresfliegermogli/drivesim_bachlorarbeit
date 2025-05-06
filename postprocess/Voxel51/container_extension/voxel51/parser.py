import os
from collections import defaultdict

import fiftyone as fo


class Parser:
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
