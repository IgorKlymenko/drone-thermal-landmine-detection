import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_run_number(folder_name):
    """Extract number after 'train' from folder name"""
    try:
        return int(folder_name.replace('train', ''))
    except ValueError:
        return None

def sanitize_filename(name):
    """Convert metric name to a valid filename"""
    return name.replace('/', '_').replace('\\', '_')

def process_runs_directory():
    runs_dir = Path('runs/classify')
    results = {}
    
    # Iterate through all train folders
    for folder in runs_dir.glob('train*'):
        if not folder.is_dir():
            continue
            
        run_number = extract_run_number(folder.name)
        if run_number is None or run_number < 10:
            # Read dropout from args.yaml for the dropout rate
            args_path = folder / 'args.yaml'
            if args_path.exists():
                with open(args_path, 'r') as f:
                    args = yaml.safe_load(f)
                    dropout = args.get('dropout', None)
            else:
                dropout = None
        else:
            dropout = None
            
        # Read results.csv (renamed to experimentN_results.csv for the committed runs)
        candidates = [folder / 'results.csv', *folder.glob('experiment*_results.csv')]
        results_path = next((p for p in candidates if p.exists()), None)
        if results_path is not None:
            df = pd.read_csv(results_path)
            metrics = df.to_dict('records')
        else:
            metrics = None
            
        results[folder.name] = {
            'run_number': run_number,
            'dropout': dropout,
            'metrics': metrics
        }
    
    return results

def plot_metrics_progression(results):
    # Filter results to only include entries with dropout values and metrics
    valid_results = {k: v for k, v in results.items() 
                    if v['dropout'] is not None and v['metrics'] is not None}
    
    if not valid_results:
        print("No valid results with dropout values found!")
        return
    
    # Create output directory for plots using absolute path
    current_dir = Path.cwd()
    plots_dir = current_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    print(f"Created plots directory at: {plots_dir}")
    
    # Group results by dropout rate and select best run for each
    dropout_groups = {}
    for folder_name, data in valid_results.items():
        dropout = data['dropout']
        if dropout not in dropout_groups:
            dropout_groups[dropout] = []
        dropout_groups[dropout].append(data)
    
    # Select best run for each dropout rate (lowest final validation loss)
    best_runs = {}
    for dropout, runs in dropout_groups.items():
        best_run = min(runs, key=lambda x: x['metrics'][-1].get('val/loss', float('inf')))
        best_runs[dropout] = best_run
    
    # Get unique dropout rates and assign colors
    dropout_rates = sorted(best_runs.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(dropout_rates)))
    dropout_colors = dict(zip(dropout_rates, colors))
    
    # Get all metric names from the first valid result
    first_metrics = next(iter(best_runs.values()))['metrics'][0]
    metric_names = [name for name in first_metrics.keys() if name != 'epoch']
    
    for metric_name in metric_names:
        plt.figure(figsize=(10, 6))
        
        for dropout, data in best_runs.items():
            metrics = data['metrics']
            
            # Extract epochs and metric values
            epochs = [m['epoch'] for m in metrics]
            values = [m[metric_name] for m in metrics]
            
            # Plot with color based on dropout rate
            plt.plot(epochs, values, 'o-', 
                    color=dropout_colors[dropout],
                    alpha=0.7,
                    label=f'Dropout={dropout}')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Progression')
        plt.grid(True)
        
        #  legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Sort legend by dropout rate
        sorted_indices = np.argsort([float(label.split('=')[1]) for label in labels])
        plt.legend([handles[i] for i in sorted_indices],
                  [labels[i] for i in sorted_indices],
                  title='Dropout Rates',
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left')
        
        plt.tight_layout()
        

        safe_metric_name = sanitize_filename(metric_name)
        output_path = plots_dir / f'{safe_metric_name}_progression.png'
        try:
            plt.savefig(str(output_path), 
                       bbox_inches='tight', 
                       dpi=300)
            print(f"Successfully saved plot to: {output_path}")
        except Exception as e:
            print(f"Error saving plot {metric_name}: {str(e)}")
        finally:
            plt.close()

if __name__ == '__main__':
    results = process_runs_directory()
    
    for folder_name, data in results.items():
        print(f"\nFolder: {folder_name}")
        print(f"Run number: {data['run_number']}")
        print(f"Dropout: {data['dropout']}")
        if data['metrics']:
            print("Metrics available in results.csv")
        print("-" * 50)
    
    # Create plots
    plot_metrics_progression(results) 