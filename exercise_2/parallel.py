import os
import csv
import time
import tifffile
from multiprocessing import Pool

from serial import load_image, segment_cells, measure_cells


def process_image_worker(image_path):
    """
    Worker function for parallel image processing.

    Runs the full pipeline on a single image. Defined at module
    level so multiprocessing can pickle it.

    Args:
        image_path (str): Path to the .tif image file.

    Returns:
        list[dict]: Measurement records for all detected cells.
    """
    image = load_image(image_path)
    masks = segment_cells(image)
    return measure_cells(masks, image_path)


def parallel_process_images(image_paths, num_workers):
    """
    Process a collection of images in parallel using multiprocessing.

    Work is distributed by image — each worker receives one image
    at a time and runs the full pipeline independently. This strategy
    was chosen because each image is processed identically and
    independently, making it a natural fit for task parallelism.
    There is no shared state between workers.

    Args:
        image_paths (list[str]): List of paths to .tif image files.
        num_workers (int): Number of parallel worker processes.

    Returns:
        list[dict]: Combined measurement records from all images.
    """
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_image_worker, image_paths)

    # flatten list of lists into a single list of records
    all_records = []
    for records in results:
        all_records.extend(records)

    return all_records


def save_results(records, path="parallel_results.csv"):
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
    image_dir = "data/DIC-C2DH-HeLa/01"
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".tif")
    ])

    print(f"Found {len(image_paths)} images")

    start = time.perf_counter()
    records = parallel_process_images(image_paths, num_workers=4)
    elapsed = time.perf_counter() - start

    print(f"Processed {len(image_paths)} images in {elapsed:.2f}s")
    print(f"Total cells detected: {len(records)}")

    save_results(records)