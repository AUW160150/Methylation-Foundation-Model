# Data Directory Structure Guide

This guide explains where to place data files and what data is automatically generated.

## Directory Structure

```
Methylation-Foundation-Model/
├── data/
│   ├── methylation/          # Place real methylation data here (from dbGaP)
│   │   ├── framingham/       # Framingham Heart Study (phs000424)
│   │   ├── goldn/           # GOLDN Study (phs000428)
│   │   └── mesa/            # MESA Study (phs000710)
│   ├── raw/                 # Raw downloaded data
│   └── processed/           # Processed training-ready data
├── checkpoints/             # Model checkpoints saved during training
│   └── best_model/         # Best performing model
├── results/                 # Evaluation results
│   ├── benchmarks/         # GUE benchmark results
│   └── plots/              # Visualization outputs
└── logs/                   # Training logs and metrics
```

## 📊 Data Sources

### **Option 1: Synthetic Data (Quick Start - No dbGaP Required)**

The **simple implementation** (`simple_foundation_Model_Training.ipynb`) automatically generates synthetic methylation data for testing:

- **Location**: Generated in memory during notebook execution
- **Size**: ~2000 sequences (configurable)
- **Format**: DNA sequences with binary methylation labels
- **Purpose**: Pipeline testing, proof-of-concept, rapid prototyping
- **Setup Required**: None - just run the notebook!

**The notebook creates:**
```python
# Binary methylation classification dataset
synthetic_methylation.csv  # 2000 samples, 200bp sequences
synthetic_age.csv          # 1000 samples for age prediction
```

These are generated on-the-fly and saved to `data/processed/` when you run Cell 6.

---

### **Option 2: Real Methylation Data (Production - Requires dbGaP Access)**

The **production implementation** (`production_methylation_foundation_model.ipynb`) requires real methylation data:

#### **Step 1: Apply for dbGaP Access**

1. Create eRA Commons account: https://public.era.nih.gov/commons/
2. Request data access at: https://dbgap.ncbi.nlm.nih.gov/
3. Submit Data Access Request (DAR) for:
   - **phs000424** (Framingham Heart Study) - 4,188 samples
   - **phs000428** (GOLDN) - 2,138 samples  
   - **phs000710** (MESA) - 4,000+ samples
4. Wait for approval (3-10 business days)

#### **Step 2: Download Data**

```bash
# Install SRA Toolkit
wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-ubuntu64.tar.gz
tar -xzf sratoolkit.current-ubuntu64.tar.gz
export PATH=$PATH:$PWD/sratoolkit.3.0.10-ubuntu64/bin

# Configure credentials
vdb-config --interactive  # Import repository key

# Download Framingham study
prefetch phs000424
vdb-dump phs000424 --output-path ./data/methylation/framingham
```

#### **Step 3: Place Data in Correct Location**

Place downloaded IDAT files and metadata in:
```
data/methylation/framingham/
├── idat/                    # Illumina IDAT files
│   ├── sample_001_Red.idat
│   ├── sample_001_Grn.idat
│   └── ...
└── metadata.csv            # Sample metadata (age, sex, tissue)
```

---

## 📁 Placeholder Directories

The following empty directories have been created with `.gitkeep` files:

- `data/methylation/` - For real methylation data (dbGaP)
- `data/raw/` - For raw downloaded data
- `data/processed/` - For processed datasets (synthetic data will be saved here)
- `checkpoints/` - For model checkpoints
- `results/benchmarks/` - For evaluation results
- `results/plots/` - For visualization outputs
- `logs/` - For training logs

**Note**: These directories are tracked in git but remain empty until you add data or run training.

---

## 🚀 Quick Start Guide

### **New User (Noah) - Start Here!**

**Option A: Quick Testing (5 minutes)**
```bash
# No data download needed!
jupyter notebook simple_foundation_Model_Training.ipynb
# Run all cells - synthetic data is generated automatically
```

**Option B: Production Setup (1-2 weeks)**
```bash
# 1. Apply for dbGaP access (wait for approval)
# 2. Download data using SRA Toolkit
# 3. Place data in data/methylation/
# 4. Run production notebook
jupyter notebook production_methylation_foundation_model.ipynb
```

---

## 📊 Data Validation

Before training, validate your data:

```bash
# For real methylation data
python validate_methylation_data.py \
    --data-dir ./data/methylation/framingham \
    --verbose \
    --export-report validation_report.txt
```

Expected output:
```
✅ VALIDATION PASSED
Data is ready for training!

Statistics:
  num_samples: 4188
  num_probes: 485512
  beta_mean: 0.517
  missing_rate: 0.5%
```

---

## 💾 Expected Data Sizes

### Synthetic Data
- `synthetic_methylation.csv`: ~500 KB
- `synthetic_age.csv`: ~250 KB
- Total: < 1 MB

### Real Data (Framingham)
- Raw IDAT files: ~50 GB
- Processed CSV: ~5 GB
- Model checkpoints: ~500 MB - 2 GB

### Results
- Benchmark results: < 10 MB
- Training logs: < 100 MB
- Plots: < 5 MB

---

## 🔒 Data Privacy & Security

**Important**: 
- **Never commit real methylation data** to GitHub (it's controlled-access data)
- The `.gitignore` file excludes `data/` directory by default
- If you uncomment data exclusions in `.gitignore`, synthetic data in `data/processed/` will also be ignored
- Always verify with `git status` before pushing

---

## ❓ Troubleshooting

**Q: Where do I put my downloaded data?**  
A: Place it in `data/methylation/[study_name]/` where study_name matches the dbGaP accession (framingham, goldn, mesa)

**Q: Do I need real data to test the pipeline?**  
A: No! Use `simple_foundation_Model_Training.ipynb` which generates synthetic data automatically.

**Q: Why are the data directories empty?**  
A: By design - you add data as needed. The `.gitkeep` files ensure directories exist but remain empty until you populate them.

**Q: Can I use my own methylation data?**  
A: Yes! Format it according to the validation requirements:
- Required columns: `probe_id`, `beta_value`, `chr`, `position`
- Beta values in range [0, 1]
- See `validate_methylation_data.py` for full requirements

---

## 📝 Summary

| Implementation | Data Source | Setup Time | Best For |
|----------------|-------------|------------|----------|
| **Simple** | Synthetic (auto-generated) | 0 minutes | Testing, Learning |
| **Production** | Real dbGaP data | 1-2 weeks | Research, Publication |

**Recommendation for Noah**: Start with the simple implementation to understand the pipeline, then transition to production with real data when dbGaP access is approved.
