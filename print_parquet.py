#!/usr/bin/env python3

import pandas as pd
import sys
from pathlib import Path
import argparse


def print_parquet_data(parquet_file: str, max_rows: int = 10, show_all_columns: bool = False):
    """Print contents of a Parquet file with formatted output."""
    
    if not Path(parquet_file).exists():
        print(f"Error: File '{parquet_file}' not found")
        return
    
    print(f"\n{'='*80}")
    print(f"Parquet File: {parquet_file}")
    print(f"{'='*80}\n")
    
    # Load the parquet file
    df = pd.read_parquet(parquet_file)
    
    # Basic information
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    
    # Display settings for better formatting
    pd.set_option('display.max_columns', None if show_all_columns else 10)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    # Show first rows
    print(f"\nFirst {min(max_rows, len(df))} rows:")
    print("-" * 80)
    print(df.head(max_rows).to_string())
    
    if len(df) > max_rows:
        print(f"\n... ({len(df) - max_rows} more rows)")
    
    # Show statistics for numeric columns
    print(f"\n{'='*80}")
    print("Statistics for numeric columns:")
    print("-" * 80)
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()
            
            print(f"\n{col}:")
            print(f"  Non-null: {non_null:,} ({non_null/len(df)*100:.1f}%)")
            print(f"  Null: {null_count:,} ({null_count/len(df)*100:.1f}%)")
            
            if non_null > 0:
                print(f"  Min: {df[col].min():.2f}")
                print(f"  Max: {df[col].max():.2f}")
                print(f"  Mean: {df[col].mean():.2f}")
                print(f"  Median: {df[col].median():.2f}")
                print(f"  Std: {df[col].std():.2f}")
    
    # Time-based analysis if time column exists
    if 'time' in df.columns:
        print(f"\n{'='*80}")
        print("Time-based information:")
        print("-" * 80)
        
        try:
            # Ensure time is datetime
            if df['time'].dtype == 'object':
                df['time'] = pd.to_datetime(df['time'])
            
            duration = df['time'].max() - df['time'].min()
            print(f"  Start: {df['time'].min()}")
            print(f"  End: {df['time'].max()}")
            print(f"  Duration: {duration}")
            print(f"  Sampling rate: {len(df) / duration.total_seconds():.2f} Hz")
        except Exception as e:
            print(f"  Could not parse time information: {e}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Display contents of a Parquet file')
    parser.add_argument('file', help='Path to the parquet file')
    parser.add_argument('--rows', '-r', type=int, default=10,
                        help='Number of rows to display (default: 10)')
    parser.add_argument('--all-columns', '-a', action='store_true',
                        help='Show all columns (default: truncate wide tables)')
    parser.add_argument('--info', '-i', action='store_true',
                        help='Show only file info without data')
    
    args = parser.parse_args()
    
    if not args.file:
        # List available parquet files if no file specified
        parquet_files = list(Path().rglob("*.parquet"))
        if parquet_files:
            print("Available Parquet files:")
            for f in parquet_files:
                print(f"  - {f}")
            print("\nUsage: python print_parquet.py <file.parquet>")
        else:
            print("No Parquet files found in current directory")
        sys.exit(1)
    
    if args.info:
        # Just show basic info
        df = pd.read_parquet(args.file)
        print(f"File: {args.file}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    else:
        print_parquet_data(args.file, args.rows, args.all_columns)


if __name__ == "__main__":
    main()