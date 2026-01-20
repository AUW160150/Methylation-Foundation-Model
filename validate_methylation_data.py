#!/usr/bin/env python3
"""
Methylation Data Validation Script
===================================

Validates methylation data format and quality before training.

Usage:
    python validate_methylation_data.py --data-dir ./data/methylation
    python validate_methylation_data.py --data-file methylation_data.csv --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class MethylationDataValidator:
    """Validates methylation data for training"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def log(self, message: str, level: str = "INFO"):
        """Log message"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            prefix = {
                "INFO": "ℹ️",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "SUCCESS": "✅"
            }.get(level, "")
            print(f"{prefix} {message}")
    
    def validate_file_format(self, file_path: Path) -> bool:
        """Validate file format and structure"""
        self.log(f"Validating file format: {file_path}", "INFO")
        
        # Check file exists
        if not file_path.exists():
            self.errors.append(f"File not found: {file_path}")
            return False
        
        # Check file extension
        valid_extensions = ['.csv', '.tsv', '.txt', '.parquet']
        if file_path.suffix not in valid_extensions:
            self.warnings.append(
                f"Unexpected file extension: {file_path.suffix}. "
                f"Expected one of: {valid_extensions}"
            )
        
        # Try to load file
        try:
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, nrows=5)
            
            self.log(f"  Loaded {len(df)} rows (preview)", "SUCCESS")
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to load file: {e}")
            return False
    
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Validate required columns exist"""
        self.log("Validating column structure", "INFO")
        
        # Required columns
        required_columns = ['probe_id', 'chr', 'position', 'beta_value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.errors.append(
                f"Missing required columns: {missing_columns}. "
                f"Found columns: {list(df.columns)}"
            )
            return False
        
        # Optional but recommended columns
        optional_columns = ['tissue', 'sample_id', 'age', 'sex']
        found_optional = [col for col in optional_columns if col in df.columns]
        missing_optional = [col for col in optional_columns if col not in df.columns]
        
        if found_optional:
            self.log(f"  Found optional columns: {found_optional}", "SUCCESS")
        if missing_optional:
            self.warnings.append(
                f"Missing optional columns: {missing_optional}. "
                "These may improve model performance."
            )
        
        return True
    
    def validate_beta_values(self, df: pd.DataFrame) -> bool:
        """Validate methylation beta values"""
        self.log("Validating beta values", "INFO")
        
        beta_col = 'beta_value'
        if beta_col not in df.columns:
            self.errors.append(f"Column '{beta_col}' not found")
            return False
        
        beta_values = df[beta_col].dropna()
        
        # Check range [0, 1]
        out_of_range = (beta_values < 0) | (beta_values > 1)
        n_out_of_range = out_of_range.sum()
        
        if n_out_of_range > 0:
            pct = 100 * n_out_of_range / len(beta_values)
            if pct > 5:
                self.errors.append(
                    f"{n_out_of_range} ({pct:.2f}%) beta values out of range [0,1]"
                )
                return False
            else:
                self.warnings.append(
                    f"{n_out_of_range} ({pct:.2f}%) beta values out of range [0,1]. "
                    "These will be filtered."
                )
        
        # Check for missing values
        n_missing = df[beta_col].isna().sum()
        if n_missing > 0:
            pct = 100 * n_missing / len(df)
            self.warnings.append(
                f"{n_missing} ({pct:.2f}%) missing beta values. "
                "These samples will be excluded."
            )
        
        # Calculate statistics
        self.stats['beta_mean'] = beta_values.mean()
        self.stats['beta_std'] = beta_values.std()
        self.stats['beta_min'] = beta_values.min()
        self.stats['beta_max'] = beta_values.max()
        
        self.log(f"  Beta value range: [{self.stats['beta_min']:.3f}, {self.stats['beta_max']:.3f}]", "SUCCESS")
        self.log(f"  Mean ± SD: {self.stats['beta_mean']:.3f} ± {self.stats['beta_std']:.3f}", "INFO")
        
        return True
    
    def validate_genomic_coordinates(self, df: pd.DataFrame) -> bool:
        """Validate chromosome and position information"""
        self.log("Validating genomic coordinates", "INFO")
        
        # Check chromosome format
        if 'chr' not in df.columns:
            self.errors.append("Column 'chr' not found")
            return False
        
        chr_values = df['chr'].unique()
        
        # Expected chromosomes (human)
        expected_chrs = (
            [f'chr{i}' for i in range(1, 23)] + 
            ['chrX', 'chrY', 'chrM']
        )
        
        unexpected_chrs = [c for c in chr_values if c not in expected_chrs]
        if unexpected_chrs:
            self.warnings.append(
                f"Unexpected chromosome names: {unexpected_chrs[:5]}. "
                f"Expected format: chr1, chr2, ..., chrX, chrY, chrM"
            )
        
        # Check position values
        if 'position' not in df.columns:
            self.errors.append("Column 'position' not found")
            return False
        
        positions = df['position'].dropna()
        
        if (positions < 0).any():
            self.errors.append("Found negative position values")
            return False
        
        if (positions > 300_000_000).any():
            self.warnings.append(
                "Found positions > 300Mb. Please verify these are correct."
            )
        
        self.stats['n_chromosomes'] = len(chr_values)
        self.stats['position_range'] = (positions.min(), positions.max())
        
        self.log(f"  {len(chr_values)} chromosomes found", "SUCCESS")
        self.log(f"  Position range: {positions.min():,} - {positions.max():,}", "INFO")
        
        return True
    
    def validate_probe_ids(self, df: pd.DataFrame) -> bool:
        """Validate probe IDs"""
        self.log("Validating probe IDs", "INFO")
        
        if 'probe_id' not in df.columns:
            self.errors.append("Column 'probe_id' not found")
            return False
        
        probe_ids = df['probe_id']
        
        # Check for missing probe IDs
        n_missing = probe_ids.isna().sum()
        if n_missing > 0:
            self.errors.append(f"{n_missing} missing probe IDs")
            return False
        
        # Check for duplicate probe IDs (per sample)
        if 'sample_id' in df.columns:
            duplicates = df.groupby('sample_id')['probe_id'].apply(
                lambda x: x.duplicated().sum()
            ).sum()
            if duplicates > 0:
                self.warnings.append(
                    f"{duplicates} duplicate probe_id entries found. "
                    "These may need to be resolved."
                )
        
        # Identify array type based on probe IDs
        probe_sample = probe_ids.iloc[0]
        if probe_sample.startswith('cg'):
            array_type = "Illumina 450K/EPIC"
        elif probe_sample.startswith('ch'):
            array_type = "Illumina EPIC v2"
        else:
            array_type = "Unknown"
            self.warnings.append(
                f"Unknown probe ID format: {probe_sample}. "
                "Expected 'cg' (450K/EPIC) or 'ch' (EPIC v2) prefix."
            )
        
        self.stats['n_probes'] = len(probe_ids.unique())
        self.stats['array_type'] = array_type
        
        self.log(f"  {self.stats['n_probes']:,} unique probes", "SUCCESS")
        self.log(f"  Array type: {array_type}", "INFO")
        
        return True
    
    def validate_sample_info(self, df: pd.DataFrame) -> bool:
        """Validate sample information"""
        self.log("Validating sample information", "INFO")
        
        if 'sample_id' not in df.columns:
            self.warnings.append(
                "Column 'sample_id' not found. "
                "Sample tracking recommended for multi-sample datasets."
            )
            return True
        
        sample_ids = df['sample_id'].unique()
        self.stats['n_samples'] = len(sample_ids)
        
        self.log(f"  {len(sample_ids)} unique samples", "SUCCESS")
        
        # Check tissue information
        if 'tissue' in df.columns:
            tissues = df['tissue'].unique()
            self.stats['tissues'] = list(tissues)
            self.log(f"  Tissues: {', '.join(tissues)}", "INFO")
        
        # Check demographic information
        if 'age' in df.columns:
            ages = df['age'].dropna()
            self.stats['age_range'] = (ages.min(), ages.max())
            self.log(f"  Age range: {ages.min():.1f} - {ages.max():.1f} years", "INFO")
        
        if 'sex' in df.columns:
            sex_counts = df['sex'].value_counts()
            self.log(f"  Sex distribution: {dict(sex_counts)}", "INFO")
        
        return True
    
    def check_data_quality(self, df: pd.DataFrame) -> bool:
        """Check overall data quality metrics"""
        self.log("Checking data quality", "INFO")
        
        # Check missing data rate
        missing_rate = df.isna().sum().sum() / (len(df) * len(df.columns))
        self.stats['missing_rate'] = missing_rate
        
        if missing_rate > 0.1:
            self.warnings.append(
                f"High missing data rate: {missing_rate*100:.1f}%. "
                "Consider imputation or filtering."
            )
        
        self.log(f"  Overall missing rate: {missing_rate*100:.2f}%", "INFO")
        
        # Check for batch effects (if batch column exists)
        if 'batch' in df.columns:
            batches = df['batch'].unique()
            self.log(f"  {len(batches)} batches detected. Consider batch correction.", "WARNING")
        
        return True
    
    def validate_directory(self, data_dir: Path) -> bool:
        """Validate entire directory of methylation files"""
        self.log(f"Validating directory: {data_dir}", "INFO")
        
        if not data_dir.exists():
            self.errors.append(f"Directory not found: {data_dir}")
            return False
        
        # Find all data files
        data_files = []
        for ext in ['.csv', '.tsv', '.txt', '.parquet']:
            data_files.extend(data_dir.glob(f"**/*{ext}"))
        
        if not data_files:
            self.warnings.append(
                f"No data files found in {data_dir}. "
                "Looking for: .csv, .tsv, .txt, .parquet"
            )
            return False
        
        self.log(f"  Found {len(data_files)} data files", "INFO")
        
        # Validate each file
        all_valid = True
        for file_path in tqdm(data_files, desc="Validating files"):
            if not self.validate_file(file_path):
                all_valid = False
        
        return all_valid
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate a single methylation data file"""
        self.log(f"\nValidating: {file_path.name}", "INFO")
        
        # Load data
        try:
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            self.errors.append(f"Failed to load {file_path.name}: {e}")
            return False
        
        self.log(f"  Loaded {len(df):,} rows × {len(df.columns)} columns", "INFO")
        
        # Run validation checks
        checks = [
            self.validate_columns(df),
            self.validate_beta_values(df),
            self.validate_genomic_coordinates(df),
            self.validate_probe_ids(df),
            self.validate_sample_info(df),
            self.check_data_quality(df),
        ]
        
        return all(checks)
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        # Errors
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        # Statistics
        if self.stats:
            print("\n📊 STATISTICS:")
            for key, value in self.stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # Overall status
        print("\n" + "="*70)
        if not self.errors:
            print("✅ VALIDATION PASSED")
            print("\nData is ready for training!")
        else:
            print("❌ VALIDATION FAILED")
            print(f"\nPlease fix {len(self.errors)} error(s) before training.")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} warning(s) - Review recommended but not blocking.")
        
        print("="*70)
    
    def export_report(self, output_path: Path):
        """Export validation report to file"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'errors': self.errors,
            'warnings': self.warnings,
            'statistics': self.stats,
            'status': 'PASSED' if not self.errors else 'FAILED'
        }
        
        if output_path.suffix == '.json':
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            with open(output_path, 'w') as f:
                f.write("METHYLATION DATA VALIDATION REPORT\n")
                f.write("="*70 + "\n\n")
                f.write(f"Timestamp: {report['timestamp']}\n")
                f.write(f"Status: {report['status']}\n\n")
                
                if report['errors']:
                    f.write("ERRORS:\n")
                    for error in report['errors']:
                        f.write(f"  - {error}\n")
                    f.write("\n")
                
                if report['warnings']:
                    f.write("WARNINGS:\n")
                    for warning in report['warnings']:
                        f.write(f"  - {warning}\n")
                    f.write("\n")
                
                if report['statistics']:
                    f.write("STATISTICS:\n")
                    for key, value in report['statistics'].items():
                        f.write(f"  {key}: {value}\n")
        
        print(f"\n✅ Report saved to: {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Validate methylation data for training"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--data-file',
        type=Path,
        help='Single data file to validate'
    )
    group.add_argument(
        '--data-dir',
        type=Path,
        help='Directory containing data files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--export-report',
        type=Path,
        help='Export validation report to file'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("METHYLATION DATA VALIDATOR")
    print("="*70 + "\n")
    
    # Initialize validator
    validator = MethylationDataValidator(verbose=args.verbose)
    
    # Run validation
    if args.data_file:
        success = validator.validate_file(args.data_file)
    else:
        success = validator.validate_directory(args.data_dir)
    
    # Print summary
    validator.print_summary()
    
    # Export report if requested
    if args.export_report:
        validator.export_report(args.export_report)
    
    # Exit with appropriate code
    sys.exit(0 if success and not validator.errors else 1)


if __name__ == '__main__':
    main()
