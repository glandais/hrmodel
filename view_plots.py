#!/usr/bin/env python3
"""
Simple script to verify and display generated plots.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_sample_plots():
    """Display a few sample plots to verify they were created correctly."""
    
    plots_dir = Path("output/ols/plots/files")
    
    if not plots_dir.exists():
        print(f"Plots directory not found: {plots_dir}")
        return
    
    # Get first few time series plots
    timeseries_plots = list(plots_dir.glob("*_timeseries.png"))[:3]
    error_plots = list(plots_dir.glob("*_error.png"))[:3]
    
    if not timeseries_plots:
        print("No time series plots found")
        return
    
    print(f"Found {len(list(plots_dir.glob('*.png')))} plot files")
    print("Sample plots:")
    for plot in timeseries_plots:
        print(f"  - {plot.name}")
    
    # Display information about the plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Sample Time Series Plots', fontsize=16)
    
    # Show time series plots
    for i, plot_file in enumerate(timeseries_plots):
        if i < 3:
            img = mpimg.imread(plot_file)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'{plot_file.stem}')
            axes[0, i].axis('off')
    
    # Show error plots
    for i, plot_file in enumerate(error_plots):
        if i < 3:
            img = mpimg.imread(plot_file)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f'{plot_file.stem}')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def list_all_plots():
    """List all generated plots."""
    plots_dir = Path("output")
    
    if not plots_dir.exists():
        print(f"Plots directory not found: {plots_dir}")
        return
    
    print("Generated Plots Structure:")
    print("=" * 50)
    
    for model_dir in plots_dir.iterdir():
        if model_dir.is_dir() and model_dir.name not in ['processed_data']:
            model_name = model_dir.name
            print(f"\n{model_name.upper()} Model:")
            print("-" * 30)
            
            # Check if this is a model directory
            plots_subdir = model_dir / "plots"
            if plots_subdir.exists():
                # Overall plots
                overall_plots = list(plots_subdir.glob("*.png"))
                if overall_plots:
                    print("Overall Performance:")
                    for plot in overall_plots:
                        print(f"  • {plot.name}")
                
                # File-specific plots
                files_dir = plots_subdir / "files"
                if files_dir.exists():
                    file_plots = list(files_dir.glob("*.png"))
                    
                    # Group by file
                    files = {}
                    for plot in file_plots:
                        file_prefix = plot.stem.split('_')[0]
                        if file_prefix not in files:
                            files[file_prefix] = []
                        files[file_prefix].append(plot.name)
                    
                    print("\nPer-File Time Series:")
                    for file_name, plots in sorted(files.items()):
                        print(f"  {file_name}:")
                        for plot in sorted(plots):
                            plot_type = "Time Series" if "timeseries" in plot else "Error Analysis"
                            print(f"    • {plot} ({plot_type})")
                
                # Show other model components
                print(f"\nOther {model_name.upper()} components:")
                if (model_dir / f"{model_name}_model.joblib").exists():
                    print(f"  • Model file: {model_name}_model.joblib")
                
                predictions_dir = model_dir / "predictions"
                if predictions_dir.exists():
                    pred_files = list(predictions_dir.glob("*.parquet"))
                    print(f"  • Predictions: {len(pred_files)} parquet files")

if __name__ == "__main__":
    print("HR Model - Plot Viewer")
    print("=" * 50)
    
    list_all_plots()
    
    print("\n" + "=" * 50)
    choice = input("\nWould you like to display sample plots? (y/n): ").lower()
    
    if choice == 'y':
        show_sample_plots()
    else:
        print("Plot listing complete!")