import json
import cv2
import datetime
import hashlib
import os
import shutil

import pydicom
import typer
from pathlib import Path

from tqdm import tqdm

CATEGORIES = ["infarct", "nonischaemic", "insertionpoint"]


def load_json(path):
    with open(path) as file:
        return json.load(file)


def hash_to_float(name):
    return int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16) % (10**8) / (10**8)


def init_coco_structure():
    return {
        "info": {
            "description": "LGE probability dataset",
            "url": "www.aid.mr",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "date_created": datetime.datetime.now().isoformat()
        },
        "licenses": [],
        "categories": [{"id": i + 1, "name": cat, "supercategory": ""} for i, cat in enumerate(CATEGORIES)],
        "images": [],
        "annotations": []
    }


def main(raw_coco_path: Path = r"E:\Dropbox\Work\Papers\ALGE\data\_annotations.coco.json",
         src_dicom_path: Path = r"E:\Dropbox\Work\Papers\ALGE\data",
         mosaic_path: Path = r"E:\Dropbox\Work\Papers\ALGE\data\png",
         output_path: Path = r"E:\Dropbox\Work\Papers\ALGE\data\data_processed",
         prop_val: float = 0.25,
         prop_test: float = 0.25):

    coco_data = load_json(raw_coco_path)

    train_data = init_coco_structure()
    val_data = init_coco_structure()
    test_data = init_coco_structure()
    train_data['licenses'], val_data['licenses'], test_data['licenses'] = coco_data.get('licenses', []), coco_data.get('licenses', []), coco_data.get('licenses', [])

    # Adjust category mapping
    category_mapping = {cat["id"]: (cat["name"].split("_")[0], int(cat["name"].split("_")[-1])) for cat in coco_data["categories"] if "_" in cat["name"]}

    # Iterate over each image in the raw COCO data
    for image in coco_data["images"]:
        image_file_name = image["file_name"]
        hash_value = hash_to_float(image_file_name)
        train_val_test = "test" if hash_value < prop_test else "val" if hash_value < prop_test + prop_val else "train"
        mosaic_json_path = mosaic_path / (image_file_name.split("_png")[0] + ".json")

        if mosaic_json_path.exists():
            coords_data = load_json(mosaic_json_path)
            annotations_for_image = [annotation for annotation in coco_data["annotations"] if annotation["image_id"] == image["id"]]
            for annotation in annotations_for_image:
                category_name, confidence = category_mapping.get(annotation["category_id"], ("unknown", 100))
                recoded_category_id = CATEGORIES.index(category_name) + 1
                for dcm_path, dcm_coords in coords_data["coords"].items():
                    start_x_dcm, start_y_dcm = dcm_coords["start"]
                    end_x_dcm, end_y_dcm = dcm_coords["end"]
                    if (start_x_dcm <= annotation["bbox"][0] < end_x_dcm) and (start_y_dcm <= annotation["bbox"][1] < end_y_dcm):
                        new_x = annotation["bbox"][0] - start_x_dcm
                        new_y = annotation["bbox"][1] - start_y_dcm
                        new_annotation = {
                            "id": len(train_data["annotations"] if train_val_test == "train" else val_data["annotations"] if train_val_test == "val" else test_data["annotations"]),
                            "image_id": len(train_data["images"] if train_val_test == "train" else val_data["images"] if train_val_test == "val" else test_data["images"]),
                            "category_id": recoded_category_id,
                            "bbox": [new_x, new_y, annotation["bbox"][2], annotation["bbox"][3]],
                            "area": annotation["area"],
                            "segmentation": annotation["segmentation"],
                            "iscrowd": annotation["iscrowd"],
                            "confidence": confidence
                        }
                        current_data = train_data if train_val_test == "train" else val_data if train_val_test == "val" else test_data
                        current_data["annotations"].append(new_annotation)

                        if all(img["id"] != new_annotation["image_id"] for img in current_data["images"]):
                            png_path = os.path.splitext(dcm_path)[0] + ".png"
                            new_image_entry = {
                                "id": new_annotation["image_id"],
                                "file_name": png_path,
                                "dcm_name": dcm_path,
                                "height": end_y_dcm - start_y_dcm,
                                "width": end_x_dcm - start_x_dcm,
                                "date_captured": image["date_captured"]
                            }
                            current_data["images"].append(new_image_entry)
        else:
            raise ValueError(f"Could not find JSON file for image {image_file_name} at {mosaic_json_path}")

    # Write out to new JSON files
    json_path = os.path.join(output_path, "annotations", "LGE")
    with open(os.path.join(json_path, "train.json"), "w") as train_outfile:
        json.dump(train_data, train_outfile, indent=4)
    with open(os.path.join(json_path, "val.json"), "w") as val_outfile:
        json.dump(val_data, val_outfile, indent=4)
    with open(os.path.join(json_path, "test.json"), "w") as test_outfile:
        json.dump(test_data, test_outfile, indent=4)

    # Write out DICOM files
    dicom_root = os.path.join(output_path, "LGE")
    for ds_name, ds in {'train': train_data,
                        'val': val_data,
                        'test': test_data}.items():
        print(f"{ds_name.upper()}")
        for img_row in tqdm(ds["images"]):
            src_path = os.path.join(src_dicom_path, img_row["dcm_name"])

            dcm = pydicom.dcmread(src_path)
            # rescale pixels using WindowWidth and WindowLevel
            img = dcm.pixel_array
            window_center = dcm.WindowCenter
            window_width = dcm.WindowWidth
            img_min = window_center - window_width // 2
            img_max = window_center + window_width // 2
            img = (img - img_min) / (img_max - img_min)
            img = (img * 255).astype('uint8')

            dest_path = os.path.join(dicom_root, ds_name, img_row["file_name"])
            if not os.path.exists(dest_path):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                # save img as a png
                cv2.imwrite(dest_path, img)



if __name__ == "__main__":
    typer.run(main)
