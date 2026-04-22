import os
import requests
import pandas as pd
import numpy as np

# load API key from environment variable to avoid hardcoding credentials
API_KEY = os.getenv("FIRMS_API_KEY")

# region bounding box: Yucatan Peninsula (lat/lon)
# selected because it has documented fire activity during dry season
AREA = "-91.5,17.5,-87.5,21.5"  # min_lon, min_lat, max_lon, max_lat
DATE = "2024-04-01"
DAYS = "5"
SENSOR = "VIIRS_SNPP_SP"


def fetch_hotspots(api_key, area, date, days, sensor):
    """
    Fetch fire hotspot detections from NASA FIRMS API.

    Uses the area endpoint to retrieve hotspot data for a specific
    geographic region and time window. Data is returned in CSV format
    and parsed into a DataFrame.

    Filtering criteria:
        - Region: Yucatan Peninsula (bounding box)
        - Sensor: VIIRS SNPP Near Real Time
        - Date: 2024-04-01, 10-day window
        - Variables used: latitude, longitude, frp (fire radiative power),
          brightness, acq_date

    Args:
        api_key (str): NASA FIRMS API key.
        area (str): Bounding box as "min_lon,min_lat,max_lon,max_lat".
        date (str): Start date in YYYY-MM-DD format.
        days (str): Number of days to retrieve.
        sensor (str): Sensor name (e.g. VIIRS_SNPP_NRT).

    Returns:
        pandas.DataFrame: Hotspot detections with columns including
            latitude, longitude, frp, brightness, acq_date.
    """
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{sensor}/{area}/{days}/{date}"
    print(f"Fetching hotspots from NASA FIRMS...")
    print(f"  Sensor: {sensor}")
    print(f"  Area: {area}")
    print(f"  Date: {date} ({days} days)")

    response = requests.get(url)
    response.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    print(f"  Retrieved {len(df)} hotspot detections")
    return df


def hotspots_to_grid(df, grid_size=100):
    """
    Map hotspot detections onto a regular 2D grid.

    Latitude and longitude coordinates are discretized into grid indices
    using min-max normalization. Each cell in the grid that contains at
    least one hotspot detection is marked as a potential ignition point.
    Fire radiative power (frp) is averaged per cell when multiple
    detections fall in the same cell.

    Args:
        df (pandas.DataFrame): Hotspot detections from fetch_hotspots.
        grid_size (int): Number of cells per dimension (grid is square).

    Returns:
        tuple:
            - grid (numpy.ndarray): 2D array of shape (grid_size, grid_size)
              with 1 where hotspots exist and 0 elsewhere.
            - frp_grid (numpy.ndarray): 2D array with average FRP per cell.
    """
    grid = np.zeros((grid_size, grid_size), dtype=int)
    frp_grid = np.zeros((grid_size, grid_size), dtype=float)

    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()

    for _, row in df.iterrows():
        # normalize coordinates to grid indices
        i = int((row["latitude"] - lat_min) / (lat_max - lat_min + 1e-9) * (grid_size - 1))
        j = int((row["longitude"] - lon_min) / (lon_max - lon_min + 1e-9) * (grid_size - 1))
        grid[i][j] = 1
        frp_grid[i][j] += row.get("frp", 1.0)

    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Ignition cells: {grid.sum()}")
    return grid, frp_grid


def save_grid(grid, frp_grid, grid_path="data/grid.npy", frp_path="data/frp_grid.npy"):
    """
    Save grid arrays to disk for use by the automaton.

    Args:
        grid (numpy.ndarray): Binary ignition grid.
        frp_grid (numpy.ndarray): FRP values per cell.
        grid_path (str): Output path for the ignition grid.
        frp_path (str): Output path for the FRP grid.
    """
    os.makedirs("data", exist_ok=True)
    np.save(grid_path, grid)
    np.save(frp_path, frp_grid)
    print(f"Grid saved to {grid_path}")
    print(f"FRP grid saved to {frp_path}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("FIRMS_API_KEY")
    if not api_key:
        raise ValueError("FIRMS_API_KEY not found. Add it to your .env file.")

    df = fetch_hotspots(api_key, AREA, DATE, DAYS, SENSOR)
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")

    grid, frp_grid = hotspots_to_grid(df, grid_size=100)
    save_grid(grid, frp_grid)