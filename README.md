# HPC Parallel Computing

Serial and parallel implementations for four scientific computing problems using Python multiprocessing and mpi4py.

---

## Table of Contents

- [How it works](#how-it-works)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Execution](#execution)
- [Notes](#notes)

---

## How it works

The project is divided into four independent exercises. Each one includes a serial baseline and one or more parallel implementations. Benchmark scripts measure execution time, speedup, and efficiency for each strategy. Results are exported to CSV and analyzed in individual notebooks.

1. **`exercise_1`** implements dense matrix multiplication with four decomposition strategies and a distributed version using mpi4py.
2. **`exercise_2`** processes microscopy images from the DIC-C2DH-HeLa dataset using Cellpose for cell segmentation and multiprocessing to parallelize by image.
3. **`exercise_3`** simulates forest fire propagation with a 2D cellular automaton initialized with real NASA FIRMS hotspot data and parallelized with mpi4py.
4. **`exercise_4`** implements K-Means on the Covertype dataset by distributing data partitions across MPI processes and aggregating partial statistics with Allreduce.

---

## Technologies

| Technology | Use |
|---|---|
| `numpy` / `scipy` | Linear algebra, sparse matrices |
| `mpi4py` | Distributed memory parallelism |
| `multiprocessing` | Shared memory parallelism |
| `cellpose` | Cell segmentation (CPSAM model, v4 default) |
| `scikit-image` | Morphological descriptor extraction |
| `scikit-learn` | Clustering metrics |
| `matplotlib` / `imageio` | Visualization and animations |
| `requests` / `pandas` | NASA FIRMS API and data analysis |
| `python-dotenv` | Environment variable management |

---

## Project Structure

```
hpc-parallel-computing/
├── exercise_1/          # Matrix multiplication
│   ├── serial.py
│   ├── parallel_rows.py
│   ├── parallel_cols.py
│   ├── parallel_blocks.py
│   ├── distributed.py
│   ├── strassen.py
│   ├── benchmark.py
│   ├── sparse.py
│   └── analysis.ipynb
├── exercise_2/          # Cell image processing
│   ├── serial.py
│   ├── parallel.py
│   ├── benchmark.py
│   ├── summary.py
│   └── analysis.ipynb
├── exercise_3/          # Forest fire automaton
│   ├── data.py
│   ├── automaton.py
│   ├── distributed.py
│   ├── benchmark.py
│   ├── visualize.py
│   └── analysis.ipynb
├── exercise_4/          # Parallel K-Means
│   ├── data.py
│   ├── serial.py
│   ├── distributed.py
│   ├── benchmark.py
│   └── analysis.ipynb
├── docs/
│   ├── assets/
│   │   ├── exercise_1/    # performance_dense.png, performance_sparse.png
│   │   ├── exercise_2/    # performance_benchmark_ex2.png, cell_morphology.png
│   │   ├── exercise_3/    # burn_progression.png, fire_spread_snapshots.png, performance_benchmark.png, fire_animation.gif
│   │   └── exercise_4/    # kmeans_convergence.png, performance_kmeans.png
│   └── report.pdf
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10 or higher
- OpenMPI or MPICH installed before installing mpi4py
- NASA FIRMS API key (free)
- DIC-C2DH-HeLa dataset downloaded
- Two sparse matrices from the SuiteSparse Matrix Collection

---

## Setup

### 1. Install MPI runtime

macOS:
```bash
brew install open-mpi
```

Linux:
```bash
sudo apt install libopenmpi-dev
```

### 2. Clone the repository and install dependencies

```bash
git clone git@github.com:usuario/hpc-parallel-computing.git
cd hpc-parallel-computing
python3.11 -m venv env-hpc
source env-hpc/bin/activate
pip install -r requirements.txt
```

### 3. Obtain the NASA FIRMS API key

1. Go to [firms.modaps.eosdis.nasa.gov/api](https://firms.modaps.eosdis.nasa.gov/api/)
2. Click on **map_key**
3. Copy the generated API key
4. Add it to the `.env` file inside `exercise_3/`

### 4. Download the DIC-C2DH-HeLa dataset

Download from [data.celltrackingchallenge.net](https://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip) and extract to `exercise_2/data/`.

### 5. Download sparse matrices

Download two matrices in Matrix Market format (`.mtx`) from [sparse.tamu.edu](https://sparse.tamu.edu/) and place them in `exercise_1/data/`:

```
exercise_1/data/
├── west0067/
│   └── west0067.mtx
└── bcsstk01/
    └── bcsstk01.mtx
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `FIRMS_API_KEY` | API key obtained from NASA FIRMS |

Create a `.env` file inside `exercise_3/`:

```
FIRMS_API_KEY=your_api_key_here
```

---

## Execution

### Exercise 1 — Matrix Multiplication

```bash
cd exercise_1
python serial.py
python parallel_rows.py
python parallel_cols.py
python parallel_blocks.py
python strassen.py
python benchmark.py
mpirun -n 4 python distributed.py
python sparse.py
```

### Exercise 2 — Cell Image Processing

```bash
cd exercise_2
python serial.py
python parallel.py
python benchmark.py
python summary.py
```

### Exercise 3 — Forest Fire Automaton

```bash
cd exercise_3
python data.py
python automaton.py
python visualize.py
mpirun -n 4 python distributed.py
mpirun -n 4 python benchmark.py
```

### Exercise 4 — Parallel K-Means

```bash
cd exercise_4
python data.py
python serial.py
mpirun -n 4 python distributed.py
mpirun -n 4 python benchmark.py
```

---

## Notes

- Random seeds are fixed at 42 across all experiments for reproducibility.
- Benchmark result CSV files (`*_results.csv`, `results.csv`) are tracked in version control. Raw dataset files remain gitignored.
- Distributed scripts must be launched with `mpirun`. Running them directly with `python` executes as a single process.
- Exercise 2 depends on Cellpose which requires significant CPU time. Running on a subset of images is acceptable for development.
- The NASA FIRMS free tier allows 5000 transactions every 10 minutes.