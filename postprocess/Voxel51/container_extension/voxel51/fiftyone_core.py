import os
import fiftyone as fo
import toml
from .basic_ds_parser import BasicDSParser
from .pointcloud_parser import PointCloudParser


def get_parser_from_writer(writer_name):
    if writer_name == "BasicDSWriterFull":
        return BasicDSParser
    elif writer_name == "PointCloudWriter":
        return PointCloudParser
    else:
        return


def get_dataset_name(data_dir, writer):
    return f"{os.path.basename(os.path.normpath(data_dir))}_{writer}"


def export_dataset(dataset, export_dir, dataset_format="fiftyone"):
    """
    Export dataset in the specified format.
    Supported formats: fiftyone, coco, yolo
    """
    print(f"Exporting dataset to {export_dir} as format: {dataset_format}")

    if dataset_format == "coco":
        dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.COCODetectionDataset,
            label_field="bbox2d"
        )
    elif dataset_format == "yolo":
        dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="bbox2d"
        )
    elif dataset_format == "fiftyone":
        dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.FiftyOneDataset
        )
    else:
        raise ValueError(f"Unsupported export format: {dataset_format}")


def main(*args, **kwargs):
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")

    with open(config_file, "r") as f:
        cfg = toml.load(f)

    fo.config.default_app_address = cfg.get("default_app_address", "0.0.0.0")

    app_config = fo.app_config.copy()
    app_config = app_config.from_dict(cfg.get("fiftyone_app_config", {}))

    print("Starting FiftyOne with the following config:")
    print(app_config)

    data_path = kwargs.get("data_path")
    writer = kwargs.get("writer")
    export_base_dir = kwargs.get("export_dir", "/tmp/replicator_out")       #Output 
    dataset_format = kwargs.get("dataset_format", "fiftyone")

    parser = get_parser_from_writer(writer)

    if parser:
        parser = parser(data_dir=data_path)
        parser.setup()

        fo_dataset_cfg = cfg.get("fiftyone_dataset_config", {})
        if not fo_dataset_cfg.get("name"):
            fo_dataset_cfg["name"] = get_dataset_name(data_path, writer)

        dataset_name = fo_dataset_cfg["name"]

        if fo.dataset_exists(dataset_name):
            print("Dataset already exists. Loading from memory.")
            dataset = fo.load_dataset(dataset_name)
        else:
            samples = parser.parse_dataset()
            dataset = fo.Dataset(**fo_dataset_cfg)
            dataset.add_samples(samples)

        # ✅ Create a new folder for the dataset export
        export_dir = os.path.join(export_base_dir, dataset_name)
        os.makedirs(export_dir, exist_ok=True)

        # ✅ Export dataset in user-defined or default format
        export_dataset(dataset, export_dir, dataset_format)

        # Optionally launch the app                         
        #sess = fo.launch_app(dataset)
        #sess.wait()

    else:
        print("Parser not found or unsupported writer.")
        exit(1)