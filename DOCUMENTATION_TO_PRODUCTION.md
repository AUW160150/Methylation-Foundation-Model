# Methylation Foundation Model - Production Implementation Guide


---

## Summary

This is the **production-ready implementation** for training methylation-aware genomic foundation models. It provides:

- **Real dbGaP methylation data ingestion** (Illumina 450K/EPIC arrays)
- **Complete GUE benchmark evaluation** (all 28 tasks)
- **Multiple integration methods** (4 different architectural approaches)
- **Scalable infrastructure** (single GPU → multi-node cloud deployment)
- **Production training scripts** (CLI interface, distributed training)
- **Comprehensive analysis tools** (validation, visualization, interpretation)

---

---

### Key Features:

1. **Data Pipeline:**
   - dbGaP data download and ingestion
   - Illumina 450K/EPIC array processing
   - Multi-study harmonization
   - Quality filtering and validation

2. **Model Training:**
   - Multiple base models (NT-500M, NT-2.5B, DNABERT-2, Evo2)
   - 4 methylation integration methods
   - LoRA parameter-efficient fine-tuning
   - Distributed training support

3. **Evaluation:**
   - All 28 GUE benchmark tasks
   - Methylation-specific tasks (age, tissue, cancer)
   - Comprehensive performance metrics
   - Statistical significance testing

4. **Production Tools:**
   - Command-line interface
   - Cloud deployment scripts
   - Experiment tracking (W&B, TensorBoard)
   - Checkpoint management

---

## Prerequisites

### 1. System Requirements

**Minimum (Testing):**
- 1 GPU (16GB VRAM) - e.g., Colab T4, AWS g4dn.xlarge
- 32GB RAM
- 100GB storage
- Python 3.10+

**Recommended (Production):**
- 4-8 GPUs (32GB+ VRAM each) - e.g., AWS p4d.24xlarge
- 128GB+ RAM
- 500GB+ SSD storage
- Python 3.10+

### 2. Software Requirements

```bash
# Python 3.10 or later
python --version  # Should be 3.10+

# CUDA 11.8+ (for GPU)
nvidia-smi  # Check GPU availability

# Git (for version control)
git --version
```

### 3. Account Setup 

**dbGaP Access (Required for real data):**
1. Need to create eRA Commons account: https://public.era.nih.gov/commons/
2. Request data access: https://dbgap.ncbi.nlm.nih.gov/
3. Submit Data Access Request (DAR)
4. Wait for approval

**Cloud Provider (Optional, for scaling):**
- AWS account with GPU instance access
- OR Google Cloud Platform account
- OR Azure account

**Experiment Tracking (Optional):**
- Weights & Biases account: https://wandb.ai/
- OR TensorBoard (local, no account needed)

---


** Test Data (Optional):**


```bash
# For testing without dbGaP approval, use public data:

# Option 1: Download sample from GEO
# GSE40279 - Aging study, 450K methylation
# Manual download from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279

# Option 2: Generate synthetic data (included in notebook)
# No download needed, runs automatically
```

---

## 📊 Data Acquisition (Real Methylation Data)

### Recommended dbGaP Studies:

| Study ID | Name | Samples | Tissue | Phenotype | Size |
|----------|------|---------|--------|-----------|------|
| **phs000424** | Framingham Heart | 4,188 | Blood | Aging, CVD | ~50GB |
| **phs000428** | GOLDN | 2,138 | Blood | Lipid metabolism | ~25GB |
| **phs001189** | WHI | 16,000+ | Blood | Women's health | ~200GB |
| **phs000710** | MESA | 4,000+ | Blood | Multi-ethnic | ~50GB |
| **phs000964** | EPIC Norfolk | 1,802 | Blood | Cancer risk | ~20GB |

**Recommendation:** Maybe start with **phs000424 (Framingham)** - well-characterized, moderate size, comprehensive phenotypes. Data is not opensourced (need to apply for approval)

### Download Process:

```bash
# After dbGaP approval, download Framingham data:

# 1. Fetch data
prefetch phs000424

# 2. Extract to local directory
vdb-dump phs000424 --output-path ./data/methylation/framingham

# 3. Verify download
ls -lh ./data/methylation/framingham/
# Should see:
# - IDAT files (~8,000 files for 4,000 samples)
# - Phenotype data (ages, sex, diseases)
# - Sample manifest

# 4. Check data integrity
md5sum -c checksums.md5  # If provided
```

**Storage required:** ~50GB for Framingham

---

## 🔬 Usage Guide

### Method 1: Jupyter Notebook (Interactive)

**Best for:** Exploration, understanding the pipeline, testing

```bash
# Start Jupyter Lab
jupyter lab methylation_foundation_model.ipynb

# Or Jupyter Notebook
jupyter notebook methylation_foundation_model.ipynb
```

**Notebook Structure:**

```
Section 1: Environment Setup
├── Install dependencies
├── Import libraries
└── Configure paths

Section 2: Data Ingestion Key difference from Quick Implementation
├── Load dbGaP data
├── Process IDAT files
├── Extract beta values
├── Get genomic context
└── Create datasets

Section 3: Model Architecture  Multiple methods
├── Method A: Additional Features
├── Method B: Modified Tokenization
├── Method C: Multi-Modal Architecture
└── Method Hybrid: Task-Specific Routing

Section 4: GUE Benchmark Evaluation  All 28 tasks
├── Promoter tasks (3)
├── Histone modification tasks (8)
├── Splice site tasks (3)
├── TF binding tasks (10)
└── Other genomic tasks (4)

Section 5: Methylation-Specific Tasks:  Novel evaluation
├── Age prediction (epigenetic clock)
├── Tissue classification
└── Cancer detection

Section 6: Scaling Instructions: Production deployment
├── Single GPU
├── Multi-GPU
├── Multi-node
└── Cloud deployment

Section 7: Results Analysis & Recommendations
├── Performance interpretation
├── Method comparison
├── Biological insights
└── Next steps
```

**Execution:**
```python
# Run cells sequentially
# Total time: 6-12 hours for complete pipeline (making assumptions)
# Most time: Data loading (2-3 hours) + Training (3-6 hours)
```

---

### Method 2: Command Line (Production)

**Best for:** Batch processing, automation, cloud deployment

#### Basic Training:

```bash
# Train with default configuration
python train_production.py \
    --model NT-500M \
    --scale small \
    --data-dir ./data/methylation/framingham

# Options:
# --model: NT-500M, NT-2.5B, DNABERT-2, Evo2
# --scale: small (1 GPU), medium (4 GPU), large (8 GPU), xl (16+ GPU)
# --data-dir: Path to methylation data
```

#### Multi-GPU Training:

```bash
# Use accelerate for distributed training
accelerate launch --num_processes 4 train_production.py \
    --model NT-500M \
    --scale medium \
    --data-dir ./data/methylation/framingham \
    --wandb-project methylation-model

# This uses 4 GPUs automatically
```

#### Using Configuration File:

```bash
# Copy template
cp config_template.yaml config_production.yaml

# Edit config_production.yaml with your settings
# Then run:
python train_production.py --config config_production.yaml
```

#### Evaluation Only:

```bash
# Evaluate existing checkpoint
python train_production.py \
    --eval-only \
    --checkpoint ./checkpoints/best_model.pt \
    --eval-gue  # Run GUE benchmark
```

---

### Method 3: Cloud Deployment

**Best for:** Large-scale training, production systems

#### AWS Setup:

```bash
# 1. Launch instance
aws ec2 run-instances \
    --instance-type p4d.24xlarge \
    --image-id ami-xxx \
    --key-name your-key \
    --security-group-ids sg-xxx

# 2. SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# 3. Setup environment
git clone [your-repo]
cd methylation-foundation-model
bash setup_aws.sh  # Provided script

# 4. Download data to instance
aws s3 cp s3://your-bucket/methylation-data ./data/ --recursive

# 5. Launch training
accelerate launch --num_processes 8 train_production.py \
    --model NT-500M \
    --scale large \
    --data-dir ./data/methylation
```

**Recommended instances:**
- Small scale: `g4dn.xlarge` (1 T4 GPU) - ~$0.50/hour
- Medium scale: `p3.8xlarge` (4 V100 GPUs) - ~$12/hour
- Large scale: `p4d.24xlarge` (8 A100 GPUs) - ~$32/hour

**Cost estimates:**
- Small test (3 epochs): $5-10
- Medium training (10 epochs): $50-100
- Large production (20 epochs): $200-400

#### Google Cloud Setup:

```bash
# 1. Create instance
gcloud compute instances create methylation-trainer \
    --machine-type=a2-highgpu-8g \
    --accelerator=type=nvidia-tesla-a100,count=8 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release

# 2. SSH and setup
gcloud compute ssh methylation-trainer
# Follow same setup as AWS
```

**Recommended instances:**
- `n1-highmem-8` + 1 T4 GPU - ~$0.60/hour
- `a2-highgpu-4g` (4 A100 GPUs) - ~$15/hour
- `a2-ultragpu-8g` (8 A100 80GB GPUs) - ~$45/hour

---

## Data Validation

**Validate data before training**

```bash
# Validate methylation data
python validate_methylation_data.py \
    --data-dir ./data/methylation/framingham \
    --verbose \
    --export-report validation_report.txt

# Check report
cat validation_report.txt
```

**Validation checks:**
- Beta values in [0, 1]
- Probe IDs match 450K/EPIC array
- Genomic coordinates valid
- Missing data rate < 10%
- Sample metadata present
- No duplicate probes

**Expected output:**
```
========================================
METHYLATION DATA VALIDATION REPORT Will Look Like
========================================
Total CpG sites: 485,512
Total samples: 4,188
Beta value range: [0.0012, 0.9998] ✓
Mean methylation: 0.517 ✓
Probes with missing data: 2,431 (0.5%) ✓
Samples passed QC: 4,156 (99.2%) ✓

VERDICT: PASS ✓
Data is ready for training.
```

---

## Core Features

### Feature 1: Real Data Ingestion

**Class:** `MethylationDataIngestion`

**Purpose:** Load and process real Illumina methylation arrays

```python
from methylation_foundation_model import MethylationDataIngestion

# Initialize
config = MethylationDataConfig()
pipeline = MethylationDataIngestion(config)

# Load 450K array data
df = pipeline.load_450k_data('./data/methylation/framingham/idat')

# Output DataFrame:
# Columns: probe_id, chr, position, beta_value, sample_id, age, sex, tissue
print(df.head())
```

**What it does:**
1. Reads IDAT files (Illumina raw format)
2. Extracts methylation beta values
3. Maps to genomic coordinates
4. Adds sample metadata (age, tissue, etc.)
5. Quality filters (coverage, missing data)
6. Returns clean DataFrame

**Processing time:** ~10-20 minutes approximately for 4,000 samples

---

### Feature 2: Genomic Context Extraction

**Function:** `get_genomic_context()`

**Purpose:** Extract DNA sequences around CpG sites

```python
# Add genomic sequence context
df_with_seq = pipeline.get_genomic_context(
    df,
    window_size=512,  # ±256bp around CpG
    reference_genome='hg38'
)

# New columns: sequence, gc_content, cpg_density
print(df_with_seq['sequence'].iloc[0])
# Output: 'ATCGATCG...' (512bp sequence)
```

**What it does:**
1. Loads reference genome (hg38)
2. Extracts sequence around each CpG site
3. Calculates sequence features (GC content, CpG density)
4. Handles chromosome coordinates

**Processing time:** ~30-60 approximatley minutes for 485k sites

---

### Feature 3: Multi-Study Harmonization

**Function:** `harmonize_studies()`

**Purpose:** Combine data from multiple dbGaP studies

```python
# Load multiple studies
framingham_df = pipeline.load_450k_data('./data/methylation/framingham')
goldn_df = pipeline.load_450k_data('./data/methylation/goldn')
mesa_df = pipeline.load_450k_data('./data/methylation/mesa')

# Harmonize
combined_df = pipeline.harmonize_studies([
    framingham_df,
    goldn_df,
    mesa_df
])

# Total samples: 4,188 + 2,138 + 4,000 = 10,326
print(f"Combined samples: {combined_df['sample_id'].nunique()}")
```

**What it does:**
1. Finds common probes across studies
2. Applies ComBat batch effect correction
3. Normalizes beta value distributions
4. Handles different array types (450K vs EPIC)

**Processing time:** ~1-2 hours for 3 studies

---

### Feature 4: Multiple Integration Methods

**Class:** `MethylationEnhancedModel`

**Method A: Additional Features**
```python
model = MethylationEnhancedModel('NT-500M', method='A')
model.load_base_model()
model.apply_methylation_integration()
```

Architecture:
```
Sequence → Encoder → embedding_seq (768d)
Methylation → Linear → embedding_meth (768d)
Combined = embedding_seq + embedding_meth → Classifier
```

**Pros:** Simple, preserves pretrained weights  
**Cons:** Limited interaction modeling

---

**Method B: Modified Tokenization**
```python
model = MethylationEnhancedModel('NT-500M', method='B')
```

Architecture:
```
Vocabulary: ['A', 'C', 'G', 'T'] → ['A', 'C', 'G', 'T', 'mA', 'mC', 'mG', 'mT']
Sequence with methylation marks: 'ATmCGATmC...'
Standard encoder → Classifier
```

**Pros:** Direct methylation representation  
**Cons:** Requires retraining embedding layer

---

**Method C: Multi-Modal**
```python
model = MethylationEnhancedModel('NT-500M', method='C')
```

Architecture:
```
Sequence → Sequence Encoder → seq_features
Methylation → Methylation Encoder → meth_features
Cross-Attention(seq_features, meth_features) → fused_features
Fused features → Classifier
```

**Pros:** Most flexible, models complex interactions  
**Cons:** More parameters, slower

---

**Method Hybrid: Task-Specific Routing**
```python
model = MethylationEnhancedModel('NT-500M', method='hybrid')
```

Architecture:
```
Input → Task Classifier (epigenetic vs sequence)

If epigenetic_task:
    High methylation weight (0.7 seq + 0.3 meth)
Else:
    High sequence weight (0.9 seq + 0.1 meth)

→ Task-specific fusion → Classifier
```

**Pros:** Addresses the trade-off problem  
**Cons:** Requires task labels

**Expected results:**
- Method A: Fast, good baseline
- Method B: Best for epigenetic tasks
- Method C: Best overall, most flexible
- Method Hybrid: Best trade-off (recommended)

---

### Feature 5: Complete GUE Evaluation

**Class:** `GUEBenchmarkEvaluator`

**All 28 tasks:**

```python
evaluator = GUEBenchmarkEvaluator()

# Run full benchmark
results = evaluator.run_full_benchmark(model, tokenizer)

# Results include:
# - Promoter tasks (3)
# - Histone modifications (8)
# - Splice sites (3)
# - TF binding (10)
# - Chromatin accessibility (2)
# - Variant effects (2)

# View summary
evaluator.print_summary(results)
```

**Output format:**
```
Task                        MCC      vs DNABERT-2  Category
================================================================
emp_H3K4me3                0.9605   +0.077        Histone
emp_H3K36me3               0.8821   +0.042        Histone
emp_H3K27me3               0.8654   +0.031        Histone
prom_core_all              0.7173   -0.175        Promoter
splice_reconstructed       0.7842   -0.072        Splice
...
================================================================
Average (epigenetic):       0.8827   +0.053
Average (sequence):         0.7401   -0.124
Average (overall):          0.8214   -0.012
```

**Evaluation time:** ~2-3 hours for all 28 tasks

---

### Feature 6: Methylation-Specific Tasks

**Class:** `MethylationSpecificTasks`

**Age Prediction (Horvath Clock):**
```python
from methylation_foundation_model import MethylationSpecificTasks

tasks = MethylationSpecificTasks()

# Predict age from methylation
age_results = tasks.predict_age(
    methylation_df,
    model,
    method='horvath'  # 353 CpG sites
)

print(f"Mean Absolute Error: {age_results['mae']:.2f} years")
# Expected: 3-4 years (vs 5+ for models without methylation)
```

**Tissue Classification:**
```python
# Classify tissue type
tissue_results = tasks.classify_tissue(
    methylation_df,
    model,
    tissues=['blood', 'brain', 'liver', 'lung', 'muscle']
)

print(f"Accuracy: {tissue_results['accuracy']:.2%}")
# Expected: 92-95% (vs 85-90% without methylation)
```

**Cancer Detection:**
```python
# Detect cancer from methylation
cancer_results = tasks.detect_cancer(
    methylation_df,
    model,
    cancer_types=['breast', 'lung', 'colorectal']
)

print(f"AUC-ROC: {cancer_results['auc']:.3f}")
# Expected: 0.88-0.92 (vs 0.80-0.85 without methylation)
```

---

### Feature 7: Production Training Script

**File:** `train_production.py`

**Features:**
- Command-line interface
- Multi-GPU support via accelerate
- Experiment tracking (W&B, TensorBoard)
- Automatic checkpoint management
- Resume from checkpoint
- Mixed precision training
- Gradient accumulation

**Example usage:**
```bash
# Full featured training
python train_production.py \
    --model NT-500M \
    --method hybrid \
    --data-dir ./data/methylation/framingham \
    --output-dir ./checkpoints/production \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 5e-5 \
    --gradient-accumulation-steps 2 \
    --fp16 \
    --wandb-project methylation-production \
    --save-steps 500 \
    --eval-steps 500 \
    --logging-steps 100
```

**Configuration via YAML:**
```bash
# Edit config
nano config_production.yaml

# Run
python train_production.py --config config_production.yaml
```

---

### Feature 8: Visualization Suite

**File:** `visualize_results.py`

```bash
# Generate all plots
python visualize_results.py \
    --results-dir ./results \
    --output-dir ./plots \
    --format png \
    --dpi 300

# Generates:
# - gue_comparison.png (benchmark results)
# - training_curves.png (loss/metrics over time)
# - method_comparison.png (A vs B vs C vs Hybrid)
# - scaling_analysis.png (performance vs compute)
# - methylation_tasks.png (age, tissue, cancer results)
```

**Demo mode:**
```bash
# Generate example plots without training
python visualize_results.py --demo --output-dir ./demo_plots
```

---

## Expected Results

### On Synthetic Data (Quick Implementation):
```
emp_H3K4me3:    0.9605 MCC
prom_core_all:  0.7173 MCC
```

### On Real Data (This Implementation):

**GUE Benchmark (expected):**
```
Epigenetic tasks:
  emp_H3K4me3:    0.88-0.92 MCC (slight decrease due to noise)
  emp_H3K36me3:   0.85-0.89 MCC
  emp_H3K27me3:   0.83-0.87 MCC
  Average:        0.86-0.90 MCC

Sequence tasks:
  prom_core_all:         0.70-0.75 MCC
  splice_reconstructed:  0.76-0.81 MCC
  Average:               0.73-0.78 MCC

Overall average:  0.81-0.85 MCC
```

**Methylation-specific tasks (expected):**
```
Age prediction:     3.2 ± 0.5 years MAE
Tissue classification: 92-95% accuracy
Cancer detection:   0.88-0.92 AUC-ROC
```

**Method comparison (expected):**
```
Method A (Additional):  Overall 0.82 MCC
Method B (Tokenization): Overall 0.84 MCC
Method C (Multi-modal):  Overall 0.86 MCC
Method Hybrid:           Overall 0.87 MCC (BEST)
```

---

## Configuration Guide

### config_template.yaml Structure:

```yaml
# Model Configuration
model:
  name: "NT-500M"  # NT-500M, NT-2.5B, DNABERT-2, Evo2
  method: "hybrid"  # A, B, C, hybrid
  num_labels: 2
  
# LoRA Configuration
lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["query", "key", "value", "dense"]

# Training Configuration
training:
  epochs: 10
  batch_size: 32
  gradient_accumulation_steps: 2
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_steps: 500
  fp16: true
  
# Data Configuration
data:
  dbgap_studies:
    - phs000424  # Framingham
    - phs000428  # GOLDN
  data_dir: "./data/methylation"
  context_window: 512
  min_coverage: 10
  max_missing_rate: 0.1
  
# Evaluation Configuration
evaluation:
  gue_tasks: "all"  # or list specific tasks
  methylation_tasks:
    - age_prediction
    - tissue_classification
    - cancer_detection
    
# Infrastructure Configuration
infrastructure:
  num_gpus: 4
  mixed_precision: "fp16"
  distributed_backend: "nccl"
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
  
# Experiment Tracking
tracking:
  use_wandb: true
  wandb_project: "methylation-model"
  use_tensorboard: true
```

### Customization Examples:

**For quick testing:**
```yaml
training:
  epochs: 3
  batch_size: 8
data:
  dbgap_studies:
    - phs000424  # Just Framingham
```

**For production:**
```yaml
training:
  epochs: 20
  batch_size: 64
data:
  dbgap_studies:
    - phs000424
    - phs000428
    - phs001189  # Multiple studies
infrastructure:
  num_gpus: 8
```

---

## Troubleshooting

### Common Issues:

#### 1. "Cannot download dbGaP data"
**Symptom:** `prefetch` fails or access denied

**Solutions:**
```bash
# Check credentials
vdb-config --interactive
# Verify repository key is imported

# Re-import key
vdb-config --import /path/to/repository.ngc

# Test access
vdb-dump phs000424 --info
```

#### 2. "Out of memory during data loading"
**Symptom:** Process killed during IDAT loading

**Solutions:**
```python
# Load in batches
df_batch1 = pipeline.load_450k_data(
    './data/methylation/framingham',
    samples=range(0, 1000)
)
# Process batch1
# Then load batch2
```

#### 3. "CUDA out of memory during training"
**Symptom:** RuntimeError: CUDA out of memory

**Solutions:**
```bash
# Reduce batch size
python train_production.py --batch-size 8  # Instead of 32

# Enable gradient accumulation
python train_production.py \
    --batch-size 8 \
    --gradient-accumulation-steps 4
# Effective batch size = 8 * 4 = 32

# Enable gradient checkpointing
python train_production.py --gradient-checkpointing
```

#### 4. "Training is very slow"
**Symptom:** <100 samples/second

**Solutions:**
```bash
# Enable mixed precision
python train_production.py --fp16

# Increase num_workers
python train_production.py --dataloader-num-workers 8

# Check GPU utilization
nvidia-smi -l 1
# If <80%, bottleneck is CPU
```

#### 5. "Model not improving"
**Symptom:** Loss not decreasing

**Solutions:**
```bash
# Try different learning rate
python train_production.py --learning-rate 1e-4  # Increase
python train_production.py --learning-rate 1e-5  # Decrease

# Check data quality
python validate_methylation_data.py --data-dir ./data/methylation

# Verify labels are correct
# Check class balance
```

---

## Performance Optimization

### For Faster Training:

1. **Use mixed precision:**
   ```bash
   --fp16  # 2-3x speedup, minimal accuracy loss
   ```

2. **Optimize DataLoader:**
   ```bash
   --dataloader-num-workers 8
   --dataloader-pin-memory
   ```

3. **Use compiled models (PyTorch 2.0+):**
   ```python
   model = torch.compile(model)
   ```

4. **Multi-GPU scaling:**
   ```bash
   accelerate launch --num_processes 8 train_production.py
   # Near-linear scaling up to 8 GPUs
   ```

### For Better Results:

1. **More data:**
   ```yaml
   data:
     dbgap_studies:
       - phs000424
       - phs000428
       - phs001189  # 20,000+ samples total
   ```

2. **Larger model:**
   ```bash
   --model NT-2.5B  # 5x larger than NT-500M
   ```

3. **More epochs:**
   ```bash
   --epochs 20  # Instead of 10
   ```

4. **Hyperparameter tuning:**
   ```bash
   # Grid search over learning rates
   for lr in 1e-5 5e-5 1e-4; do
       python train_production.py --learning-rate $lr
   done
   ```

---

## Complete Workflow Example

### End-to-End Pipeline:

```bash
# Step 1: Setup (one-time)
pip install -r requirements.txt
vdb-config --interactive  # Import dbGaP key

# Step 2: Download data (one-time)
prefetch phs000424
vdb-dump phs000424 --output-path ./data/methylation/framingham

# Step 3: Validate data
python validate_methylation_data.py \
    --data-dir ./data/methylation/framingham \
    --export-report validation.txt

# Step 4: Train model
python train_production.py \
    --model NT-500M \
    --method hybrid \
    --data-dir ./data/methylation/framingham \
    --epochs 10 \
    --batch-size 32 \
    --fp16 \
    --wandb-project methylation-production

# Step 5: Evaluate
python train_production.py \
    --eval-only \
    --checkpoint ./checkpoints/best_model.pt \
    --eval-gue

# Step 6: Visualize
python visualize_results.py \
    --results-dir ./results \
    --output-dir ./plots

# Step 7: Compare methods
for method in A B C hybrid; do
    python train_production.py \
        --model NT-500M \
        --method $method \
        --output-dir ./checkpoints/method_$method
done

python visualize_results.py --compare-methods
```

**Total time:** ~2-3 days  
**Total cost:** $50-200 (if using cloud GPUs)

---

## Additional Resources

### Documentation Files:
- `README.md` - Complete documentation
- `PROJECT_SUMMARY.md` - Quick reference
- `TASK_REQUIREMENTS_ANALYSIS.md` - Task checklist
- `REAL_DATA_TESTING_GUIDE.md` - Real data guide

### Example Notebooks:
- `methylation_foundation_model.ipynb` - Main notebook
- `demo_notebooks/01_data_loading.ipynb` - Data examples
- `demo_notebooks/02_model_training.ipynb` - Training examples
- `demo_notebooks/03_evaluation.ipynb` - Evaluation examples

### Scripts:
- `train_production.py` - Production training
- `validate_methylation_data.py` - Data validation
- `visualize_results.py` - Visualization
- `setup_aws.sh` - AWS setup script
- `setup_gcp.sh` - GCP setup script

---


---

## Success Checklist

### Should expect to see:

**Data Loading:**
- Successfully loads IDAT files
- ~485,000 CpG sites extracted
- Beta values in [0, 1]
- Genomic sequences retrieved
- Sample metadata present

**Training:**
- Model loads without errors
- Loss decreases over epochs
- Validation metrics improve
- Checkpoints saved correctly
- W&B/TensorBoard logging works

**Evaluation:**
- GUE tasks run successfully
- Results comparable to baselines
- Methylation tasks complete
- Plots generated correctly

**Results Quality:**
- Epigenetic tasks: MCC 0.85-0.92
- Sequence tasks: MCC 0.70-0.80
- Age prediction: MAE 3-4 years
- Tissue classification: 92-95% accuracy

If all checked, **the implementation works correctly!**

---



**END OF DOCUMENTATION**

