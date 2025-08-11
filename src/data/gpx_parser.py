import gpxpy
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import glob


class GPXParser:
    """Parse GPX files and convert to pandas DataFrames."""
    
    def __init__(self):
        self.namespaces = {
            'gpx': 'http://www.topografix.com/GPX/1/1',
            'gpxtpx': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1',
        }
    
    def parse_file(self, gpx_file: str) -> pd.DataFrame:
        """Parse a single GPX file into a DataFrame."""
        with open(gpx_file, 'r') as f:
            gpx = gpxpy.parse(f)
        
        data = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    row = {
                        'time': point.time,
                        'lat': point.latitude,
                        'lon': point.longitude,
                        'ele': point.elevation,
                        'hr': 0.0,
                        'cad': 0.0,
                        'power': 0.0
                    }
                    
                    # Parse extensions
                    if point.extensions:
                        for ext in point.extensions:
                            if ext.tag.endswith('TrackPointExtension'):
                                for child in ext:
                                    if child.tag.endswith('hr'):
                                        row['hr'] = float(child.text)
                                    elif child.tag.endswith('cad'):
                                        row['cad'] = float(child.text)
                            elif ext.tag.endswith('power') or ext.tag == 'power':
                                row['power'] = float(ext.text)
                    
                    data.append(row)
        
        df = pd.DataFrame(data)
        
        # Convert time to timestamp
        if 'time' in df.columns and not df.empty:
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df['timestamp'] = df['time'].apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
        
        # Ensure numeric types
        numeric_cols = ['hr', 'cad', 'power', 'ele', 'lat', 'lon']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def parse_directory(self, gpx_dir: str) -> Dict[str, pd.DataFrame]:
        """Parse all GPX files in a directory."""
        gpx_files = glob.glob(str(Path(gpx_dir) / "*.gpx"))
        results = {}
        
        for gpx_file in gpx_files:
            file_name = Path(gpx_file).stem
            try:
                df = self.parse_file(gpx_file)
                results[file_name] = df
                print(f"Parsed {file_name}: {len(df)} points")
            except Exception as e:
                print(f"Error parsing {gpx_file}: {e}")
        
        return results