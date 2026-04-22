import os
import numpy as np
import tifffile
import csv
from skimage import measure
from cellpose import models


def load_image(path):
    """
    Load a .tif microscopy image from disk.

    Args:
        path (str): Path to the .tif file.

    Returns:
        numpy.ndarray: Grayscale image array.
    """
    return tifffile.imread(path)


def segment_cells(image):
    """
    Segment cells in a microscopy image using Cellpose.

    Uses the pretrained 'cyto2' model which is optimized for
    DIC microscopy images. Cellpose returns a label matrix where
    each unique integer represents one detected cell.

    Model settings:
        - model: cyto2
        - diameter: None (auto-estimated per image)
        - channels: [0, 0] (grayscale input, no separate nuclear channel)

    Args:
        image (numpy.ndarray): Grayscale microscopy image.

    Returns:
        numpy.ndarray: Label matrix of shape (H, W) where each cell
            has a unique integer label and background is 0.
    """
    model = models.Cellpose(model_type="cyto2", gpu=False)
    # channels [0, 0] means grayscale image with no nuclear channel
    masks, _, _, _ = model.eval(image, diameter=None, channels=[0, 0])
    return masks


def measure_cells(masks, image_path):
    """
    Extract geometric descriptors for each segmented cell.

    Uses skimage.measure.regionprops to compute object-level measurements
    from the label matrix produced by Cellpose. All measurements are
    reported in pixels.

    Args:
        masks (numpy.ndarray): Label matrix from segment_cells.
        image_path (str): Source image path, used as identifier in results.

    Returns:
        list[dict]: One record per detected cell with the following fields:
            - image: source filename
            - cell_id: unique label of the cell
            - area: number of pixels in the region
            - bbox: axis-aligned bounding box (min_row, min_col, max_row, max_col)
            - major_axis_length: length of the major axis of the fitted ellipse (pixels)
            - minor_axis_length: length of the minor axis of the fitted ellipse (pixels)
    """
    props = measure.regionprops(masks)
    records = []

    for prop in props:
        records.append({
            "image": os.path.basename(image_path),
            "cell_id": prop.label,
            "area": prop.area,
            "bbox": prop.bbox,
            "major_axis_length": round(prop.major_axis_length, 4),
            "minor_axis_length": round(prop.minor_axis_length, 4),
        })

    return records


def process_image(image_path):
    """
    Run the full serial pipeline on a single image.

    Loads the image, segments cells with Cellpose, and computes
    geometric measurements for each detected object.

    Args:
        image_path (str): Path to the .tif image file.

    Returns:
        list[dict]: Measurement records for all detected cells.
    """
    image = load_image(image_path)
    masks = segment_cells(image)
    records = measure_cells(masks, image_path)
    return records


def save_results(records, path="serial_results.csv"):
    """
    Save cell measurement records to a CSV file.

    Args:
        records (list[dict]): List of cell measurement records.
        path (str): Output file path.
    """
    fieldnames = ["image", "cell_id", "area", "bbox", "major_axis_length", "minor_axis_length"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Results saved to {path}")


if __name__ == "__main__":
    # process the first sequence of images
    image_dir = "data/DIC-C2DH-HeLa/01"
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".tif")
    ])

    print(f"Found {len(image_paths)} images in {image_dir}")
    print(f"Image format: .tif")

    # inspect first image metadata
    sample = tifffile.imread(image_paths[0])
    print(f"Image shape: {sample.shape}")
    print(f"Image dtype: {sample.dtype}")

    all_records = []
    for path in image_paths:
        print(f"Processing {os.path.basename(path)}...")
        records = process_image(path)
        print(f"  Detected {len(records)} cells")
        all_records.extend(records)

    save_results(all_records)
    print(f"\nTotal cells detected: {len(all_records)}")