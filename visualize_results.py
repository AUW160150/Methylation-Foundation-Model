#!/usr/bin/env python3
"""
Methylation Model Visualization Script
======================================

Generate comprehensive visualizations of model performance.

Usage:
    python visualize_results.py --results ./results/gue_benchmarks.csv
    python visualize_results.py --results-dir ./results --output ./plots
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class ResultsVisualizer:
    """Generate visualizations for model results"""
    
    def __init__(self, output_dir: Path = Path("./plots")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_gue_comparison(self, results_df: pd.DataFrame, save_name: str = "gue_comparison.png"):
        """
        Plot comparison of GUE benchmark performance
        
        Args:
            results_df: DataFrame with columns [task, model, mcc]
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        tasks = results_df['task'].unique()
        models = results_df['model'].unique()
        
        # Plot 1: Grouped bar chart
        x = np.arange(len(tasks))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            model_data = results_df[results_df['model'] == model]
            scores = [model_data[model_data['task'] == task]['mcc'].values[0] 
                     for task in tasks]
            
            offset = (i - len(models)/2) * width + width/2
            axes[0].bar(x + offset, scores, width, label=model, alpha=0.8)
        
        axes[0].set_xlabel('Task')
        axes[0].set_ylabel('MCC Score')
        axes[0].set_title('GUE Benchmark Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(tasks, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # Plot 2: Improvement heatmap
        baseline_model = 'DNABERT-2'
        if baseline_model in models:
            pivot_data = results_df.pivot(index='task', columns='model', values='mcc')
            improvement = pivot_data.subtract(pivot_data[baseline_model], axis=0) * 100
            improvement = improvement.drop(columns=[baseline_model])
            
            sns.heatmap(improvement, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                       cbar_kws={'label': 'Improvement (%)'}, ax=axes[1])
            axes[1].set_title(f'Improvement over {baseline_model}')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    def plot_scaling_analysis(self, scaling_df: pd.DataFrame, save_name: str = "scaling_analysis.png"):
        """
        Plot how performance improves with scale
        
        Args:
            scaling_df: DataFrame with columns [scale, task, mcc, training_time]
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Performance vs Scale
        for task in scaling_df['task'].unique():
            task_data = scaling_df[scaling_df['task'] == task]
            axes[0].plot(task_data['scale'], task_data['mcc'], 
                        marker='o', label=task, linewidth=2)
        
        axes[0].set_xlabel('Training Scale')
        axes[0].set_ylabel('MCC Score')
        axes[0].set_title('Performance vs Training Scale')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # Plot 2: Training Time vs Scale
        avg_time = scaling_df.groupby('scale')['training_time'].mean()
        axes[1].bar(avg_time.index, avg_time.values, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Training Scale')
        axes[1].set_ylabel('Training Time (hours)')
        axes[1].set_title('Average Training Time by Scale')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add efficiency annotation
        for i, (scale, time) in enumerate(avg_time.items()):
            axes[1].text(i, time, f'{time:.1f}h', 
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    def plot_methylation_tasks(self, meth_df: pd.DataFrame, save_name: str = "methylation_tasks.png"):
        """
        Plot methylation-specific task results
        
        Args:
            meth_df: DataFrame with methylation task results
        """
        fig = plt.figure(figsize=(18, 5))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Age Prediction
        ax1 = fig.add_subplot(gs[0, 0])
        if 'age_prediction' in meth_df['task'].values:
            age_data = meth_df[meth_df['task'] == 'age_prediction'].iloc[0]
            models = ['Horvath\nBaseline', 'Our Model']
            maes = [age_data['baseline_mae'], age_data['mae']]
            colors = ['#A23B72', '#2E86AB']
            
            bars = ax1.bar(models, maes, color=colors, alpha=0.7)
            ax1.set_ylabel('Mean Absolute Error (years)')
            ax1.set_title('Epigenetic Age Prediction')
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, mae in zip(bars, maes):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mae:.2f}y', ha='center', va='bottom')
        
        # Plot 2: Tissue Classification
        ax2 = fig.add_subplot(gs[0, 1])
        if 'tissue_classification' in meth_df['task'].values:
            tissue_data = meth_df[meth_df['task'] == 'tissue_classification'].iloc[0]
            # Assume tissue_f1 is stored as dict in the dataframe
            tissue_f1 = eval(tissue_data['tissue_f1'])
            
            tissues = list(tissue_f1.keys())
            f1_scores = list(tissue_f1.values())
            
            ax2.barh(tissues, f1_scores, color='#06A77D', alpha=0.7)
            ax2.set_xlabel('F1 Score')
            ax2.set_title('Tissue Classification')
            ax2.set_xlim([0, 1])
            ax2.grid(axis='x', alpha=0.3)
        
        # Plot 3: Cancer Detection
        ax3 = fig.add_subplot(gs[0, 2])
        if 'cancer_detection' in meth_df['task'].values:
            cancer_data = meth_df[meth_df['task'] == 'cancer_detection'].iloc[0]
            metrics = ['AUC', 'Sensitivity', 'Specificity']
            values = [cancer_data['auc'], 
                     cancer_data['sensitivity'],
                     cancer_data['specificity']]
            
            bars = ax3.bar(metrics, values, color='#D81E5B', alpha=0.7)
            ax3.set_ylabel('Score')
            ax3.set_title('Cancer Detection')
            ax3.set_ylim([0, 1])
            ax3.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom')
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    def plot_training_curves(self, training_log: pd.DataFrame, save_name: str = "training_curves.png"):
        """
        Plot training curves (loss, metrics over time)
        
        Args:
            training_log: DataFrame with columns [epoch, train_loss, eval_loss, eval_mcc]
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Loss curves
        axes[0].plot(training_log['epoch'], training_log['train_loss'], 
                    label='Train Loss', linewidth=2, color='#2E86AB')
        axes[0].plot(training_log['epoch'], training_log['eval_loss'], 
                    label='Validation Loss', linewidth=2, color='#A23B72')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Metrics
        if 'eval_mcc' in training_log.columns:
            axes[1].plot(training_log['epoch'], training_log['eval_mcc'], 
                        label='MCC', linewidth=2, color='#06A77D')
        if 'eval_accuracy' in training_log.columns:
            axes[1].plot(training_log['epoch'], training_log['eval_accuracy'], 
                        label='Accuracy', linewidth=2, color='#D81E5B')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    def plot_method_comparison(self, methods_df: pd.DataFrame, save_name: str = "method_comparison.png"):
        """
        Compare different methylation integration methods
        
        Args:
            methods_df: DataFrame with columns [method, task, mcc]
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        pivot_data = methods_df.pivot(index='task', columns='method', values='mcc')
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu',
                   cbar_kws={'label': 'MCC Score'}, ax=ax,
                   vmin=0, vmax=1)
        
        ax.set_title('Methylation Integration Method Comparison', fontsize=14, pad=20)
        ax.set_xlabel('Integration Method', fontsize=12)
        ax.set_ylabel('Task', fontsize=12)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    def create_dashboard(self, results_dict: Dict[str, pd.DataFrame]):
        """
        Create comprehensive dashboard with all visualizations
        
        Args:
            results_dict: Dictionary with keys:
                - 'gue': GUE benchmark results
                - 'methylation': Methylation task results
                - 'scaling': Scaling analysis
                - 'training': Training curves
                - 'methods': Method comparison
        """
        print("\n" + "="*70)
        print("GENERATING VISUALIZATION DASHBOARD")
        print("="*70 + "\n")
        
        if 'gue' in results_dict:
            self.plot_gue_comparison(results_dict['gue'])
        
        if 'methylation' in results_dict:
            self.plot_methylation_tasks(results_dict['methylation'])
        
        if 'scaling' in results_dict:
            self.plot_scaling_analysis(results_dict['scaling'])
        
        if 'training' in results_dict:
            self.plot_training_curves(results_dict['training'])
        
        if 'methods' in results_dict:
            self.plot_method_comparison(results_dict['methods'])
        
        print("\n" + "="*70)
        print(f"✅ All visualizations saved to: {self.output_dir}")
        print("="*70)


def create_demo_data():
    """Create demo data for visualization testing"""
    
    # GUE benchmark results
    gue_data = pd.DataFrame({
        'task': ['prom_core_all', 'emp_H3K4me3', 'emp_H3K36me3', 'splice_reconstructed'] * 3,
        'model': ['DNABERT-2']*4 + ['NT-500M']*4 + ['Methylation Model']*4,
        'mcc': [0.892, 0.890, 0.952, 0.872,  # DNABERT-2
                0.879, 0.875, 0.941, 0.856,  # NT-500M
                0.717, 0.961, 0.958, 0.868]  # Methylation Model
    })
    
    # Methylation task results
    meth_data = pd.DataFrame({
        'task': ['age_prediction', 'tissue_classification', 'cancer_detection'],
        'mae': [3.2, np.nan, np.nan],
        'baseline_mae': [4.9, np.nan, np.nan],
        'accuracy': [np.nan, 0.94, np.nan],
        'f1_weighted': [np.nan, 0.93, np.nan],
        'tissue_f1': [np.nan, "{'blood': 0.98, 'brain': 0.92, 'liver': 0.91, 'lung': 0.93, 'muscle': 0.89}", np.nan],
        'auc': [np.nan, np.nan, 0.88],
        'sensitivity': [np.nan, np.nan, 0.85],
        'specificity': [np.nan, np.nan, 0.90],
    })
    
    # Scaling analysis
    scaling_data = pd.DataFrame({
        'scale': ['small', 'medium', 'large'] * 2,
        'task': ['prom_core_all']*3 + ['emp_H3K4me3']*3,
        'mcc': [0.70, 0.72, 0.73, 0.94, 0.96, 0.97],
        'training_time': [2, 6, 14, 2.5, 6.5, 15],
    })
    
    # Training curves
    epochs = np.arange(1, 21)
    training_data = pd.DataFrame({
        'epoch': epochs,
        'train_loss': 0.5 * np.exp(-0.2 * epochs) + 0.1,
        'eval_loss': 0.5 * np.exp(-0.15 * epochs) + 0.15,
        'eval_mcc': 1 - 0.4 * np.exp(-0.2 * epochs),
        'eval_accuracy': 1 - 0.3 * np.exp(-0.25 * epochs),
    })
    
    # Method comparison
    methods_data = pd.DataFrame({
        'method': ['A', 'B', 'C', 'Hybrid'] * 2,
        'task': ['prom_core_all']*4 + ['emp_H3K4me3']*4,
        'mcc': [0.72, 0.68, 0.75, 0.78, 0.94, 0.92, 0.96, 0.97],
    })
    
    return {
        'gue': gue_data,
        'methylation': meth_data,
        'scaling': scaling_data,
        'training': training_data,
        'methods': methods_data,
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for methylation model results"
    )
    
    parser.add_argument(
        '--results-dir',
        type=Path,
        help='Directory containing results CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./plots'),
        help='Output directory for plots'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Generate demo visualizations with synthetic data'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize visualizer
    visualizer = ResultsVisualizer(output_dir=args.output_dir)
    
    if args.demo:
        print("Generating demo visualizations...")
        results_dict = create_demo_data()
    elif args.results_dir:
        print(f"Loading results from: {args.results_dir}")
        results_dict = {}
        
        # Load available result files
        if (args.results_dir / 'gue_benchmarks.csv').exists():
            results_dict['gue'] = pd.read_csv(args.results_dir / 'gue_benchmarks.csv')
        
        if (args.results_dir / 'methylation_tasks.csv').exists():
            results_dict['methylation'] = pd.read_csv(args.results_dir / 'methylation_tasks.csv')
        
        if (args.results_dir / 'scaling_analysis.csv').exists():
            results_dict['scaling'] = pd.read_csv(args.results_dir / 'scaling_analysis.csv')
        
        if (args.results_dir / 'training_log.csv').exists():
            results_dict['training'] = pd.read_csv(args.results_dir / 'training_log.csv')
        
        if not results_dict:
            print("No result files found. Use --demo flag to generate demo visualizations.")
            return
    else:
        print("Error: Provide either --results-dir or --demo flag")
        return
    
    # Generate dashboard
    visualizer.create_dashboard(results_dict)
    
    print("\n✅ Visualization complete!")


if __name__ == '__main__':
    main()
