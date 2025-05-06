import os
import shutil
import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone import types 

# === Load Dataset from JSON ===
dataset_dir = <Weg_zum_Ausgangsdatensatz>                       #Input setzen

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.FiftyOneDataset,
)

# === Filter only 2D BBoxes with selected labels ===    
target_labels = {"automobile", "person", "road_sign"}           #Hier kann nach Labels gefiltert werden

def filter_detections(sample):      	                        #Filtert nur nach 2d Bounding Boxes. 3d Bounding Boxes können funktionieren, führten aber wiederholt zu problemen und waren für das KI-Training irrelevant.
    if sample.has_field("bbox2d"):
        detections = sample["bbox2d"].detections
        filtered_detections = [
            d for d in detections if d.label.split(",")[0].strip() in target_labels
        ]
        sample["bbox2d"].detections = filtered_detections
        sample.save()

# Apply filtering to each sample
for sample in dataset:
    filter_detections(sample)

# Filter samples that still have at least one detection
filtered_dataset = dataset.match(F("bbox2d.detections").length() > 0)

# === Set Output Root Directory ===
dataset_name = filtered_dataset.name or "FilteredDataset"
output_root = os.path.join(<Weg_zum_Outputordner>, dataset_name)        #Output Setzen
os.makedirs(output_root, exist_ok=True)




# === EXPORT 1: Images with Bounding Boxes Drawn (PNG) ===              #Exportiert die .jpgs
export_dir_bb = os.path.join(output_root, "with_bboxes")
os.makedirs(export_dir_bb, exist_ok=True)

# Check if the dataset is grouped (older version = media_type is string)
print("Media type:", dataset.media_type)
print("Group slices (raw):", dataset.group_slices)

if dataset.media_type == "group":
    # If grouped, select a slice (use the first or a known one like "image" or "rgb")
    slice_name = dataset.group_slices[0]  # Replace if needed
    image_slice = filtered_dataset.select_group_slices(slice_name)
    image_slice.compute_metadata(overwrite=True)
    image_slice.draw_labels(
        output_dir=export_dir_bb,
        label_fields="bbox2d",
        overwrite=True
    )
else:
    # If not grouped, proceed normally
    filtered_dataset.compute_metadata(overwrite=True)
    filtered_dataset.draw_labels(
        output_dir=export_dir_bb,
        label_fields="bbox2d",
        overwrite=True
    )

print(f"\n✅ Images with bounding boxes saved to: {export_dir_bb}")



# === EXPORT 2: YOLO Format ===                                   #Expotiert das YOLO Format
export_dir_yolo = os.path.join(output_root, "yolo_export")
images_root = os.path.join(export_dir_yolo, "images")
labels_root = os.path.join(export_dir_yolo, "labels")

os.makedirs(images_root, exist_ok=True)
os.makedirs(labels_root, exist_ok=True)

# Split into train/val
filtered_dataset.shuffle()
train_dataset = filtered_dataset[:int(0.8 * len(filtered_dataset))]
val_dataset = filtered_dataset[int(0.8 * len(filtered_dataset)):]

# === Label mapping ===
class_labels = sorted(target_labels)

def export_yolo(ds, split):
    images_out = os.path.join(images_root, split)
    labels_out = os.path.join(labels_root, split)

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # Clean labels + copy images
    for sample in ds:
        if sample.has_field("bbox2d"):
            detections = sample.bbox2d
            for det in detections.detections:
                if det.label:
                    det.label = det.label.split(",")[0].strip()
            sample.save()

        # Copy image manually
        src_img = sample.filepath
        filename = os.path.basename(src_img)
        dst_img = os.path.join(images_out, filename)
        shutil.copy2(src_img, dst_img)

    # Export labels to a temp folder
    temp_export = os.path.join(export_dir_yolo, f"temp_labels_{split}")
    os.makedirs(temp_export, exist_ok=True)

    ds.export(
        export_dir=temp_export,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="bbox2d",
        split=split,
        classes=class_labels,
        export_media=False,
        overwrite=True,
    )

    # Move .txt files from temp_export/labels/split -> labels/train or labels/val
    nested_label_dir = os.path.join(temp_export, "labels", split)
    for file in os.listdir(nested_label_dir):
        if file.endswith(".txt"):
            shutil.move(os.path.join(nested_label_dir, file), os.path.join(labels_out, file))

    shutil.rmtree(temp_export)

export_yolo(train_dataset, "train")
export_yolo(val_dataset, "val")

print(f"\n✅ YOLOv8 format exported to: {export_dir_yolo}")

# === Generate dataset.yaml for YOLOv8 ===                      
names_block = "\n".join([f"  {i}: {name}" for i, name in enumerate(class_labels)])

yaml_content = f"""\
path: {export_dir_yolo}
train: images/train
val: images/val
test:  # optional
names:
{names_block}
"""

yaml_path = os.path.join(export_dir_yolo, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"\n✅ YOLOv8-compatible dataset.yaml saved to: {yaml_path}")