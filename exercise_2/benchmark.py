import os
import time
import csv
from multiprocessing import cpu_count

from serial import load_model, process_image
from parallel import parallel_process_images


def benchmark(image_paths, workers_list):
    """
    Run serial and parallel pipelines and measure execution times.

    The serial pipeline processes images one by one. The parallel
    pipeline is tested with each number of workers in workers_list.
    Results are returned as a list of records for CSV export.

    Args:
        image_paths (list[str]): List of paths to .tif image files.
        workers_list (list[int]): List of worker counts to evaluate.

    Returns:
        list[dict]: Benchmark records with strategy, workers and time.
    """
    results = []

    # serial benchmark — model loaded once and reused across all images
    print("Running serial pipeline...")
    model = load_model()
    start = time.perf_counter()
    for path in image_paths:
        process_image(path, model)
    serial_time = time.perf_counter() - start
    print(f"  serial  time={serial_time:.4f}s")
    results.append({
        "strategy": "serial",
        "num_workers": 1,
        "num_images": len(image_paths),
        "time_seconds": round(serial_time, 6),
        "speedup": 1.0,
    })

    # parallel benchmark for each worker count
    for num_workers in workers_list:
        print(f"Running parallel pipeline with {num_workers} workers...")
        start = time.perf_counter()
        parallel_process_images(image_paths, num_workers)
        parallel_time = time.perf_counter() - start
        speedup = round(serial_time / parallel_time, 4)
        print(f"  parallel  workers={num_workers}  time={parallel_time:.4f}s  speedup={speedup}x")
        results.append({
            "strategy": "parallel",
            "num_workers": num_workers,
            "num_images": len(image_paths),
            "time_seconds": round(parallel_time, 6),
            "speedup": speedup,
        })

    return results


def save_results(records, path="benchmark_results.csv"):
    """
    Save benchmark results to a CSV file.

    Args:
        records (list[dict]): List of benchmark records.
        path (str): Output file path.
    """
    fieldnames = ["strategy", "num_workers", "num_images", "time_seconds", "speedup"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    image_dir = "data/DIC-C2DH-HeLa/01"
    all_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".tif")
    ])
    image_paths = all_paths[:10]

    print(f"Available CPU cores: {cpu_count()}")
    print(f"Found {len(all_paths)} images (benchmarking first {len(image_paths)})\n")

    # test with increasing number of workers
    workers_list = [2, 4, 8]

    records = benchmark(image_paths, workers_list)
    save_results(records)