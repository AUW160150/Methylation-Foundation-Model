# Methylation Foundation Model

Production-ready pipeline for building and evaluating methylation-aware genomic foundation models.

---

## 🚀 **Quick Start for Noah**

**Want to test immediately? (5 minutes)**

1. Open [`simple_foundation_Model_Training.ipynb`](simple_foundation_Model_Training.ipynb) in Google Colab
2. Runtime > Change runtime type > **T4 GPU**
3. Runtime > **Run all**
4. ✅ Done! Results in ~20 minutes (synthetic data auto-generated)

**No setup, no data download, no configuration needed!**

📖 [Read the Simple Implementation Guide](DOCUMENTATION_TO_SIMPLE_SCRIPT.md) for details.

---

**Need production setup with real data?**
- 📖 Read [DATA_GUIDE.md](DATA_GUIDE.md) - explains data placement and synthetic vs real data
- 📖 Read [Production Guide](DOCUMENTATION_TO_PRODUCTION.md) - dbGaP access and cloud deployment
- 📓 Use `production_methylation_foundation_model.ipynb` with real methylation data

---

## Project Overview

This fine-tunes pre-trained genomic foundation models (Nucleotide Transformer, DNABERT-2, Evo2) with DNA methylation data to improve performance on epigenetic prediction tasks.

**Key Features:**
- Complete data ingestion pipeline from dbGaP
- Multiple methylation integration methods
- GUE benchmark evaluation suite
- Methylation-specific task evaluation
- Production-ready scaling for cloud deployment
- Comprehensive analysis and visualization

**Key Insight:** The model excels at epigenetic tasks where methylation patterns provide direct signal, but shows trade-offs on pure sequence tasks.

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/AUW160150/Methylation-Foundation-Model
cd methylation-model

# Create virtual environment
conda create -n methylation python=3.10
conda activate methylation

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

#### Option A: Using dbGaP Data (Recommended)

Follow these steps to access controlled methylation data:

1. **Create eRA Commons Account**
   - Visit: https://public.era.nih.gov/commons/
   - Complete registration

2. **Request Data Access**
   - Go to: https://dbgap.ncbi.nlm.nih.gov/
   - Select study (e.g., phs000424 - Framingham Heart Study)
   - Submit Data Access Request (DAR)
   - Approval time: 3-10 business days

3. **Setup SRA Toolkit**
   ```bash
   wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-ubuntu64.tar.gz
   tar -xzf sratoolkit.current-ubuntu64.tar.gz
   export PATH=$PATH:$PWD/sratoolkit.3.0.10-ubuntu64/bin
   vdb-config --interactive  # Import repository key
   ```

4. **Download Data**
   ```bash
   prefetch phs000424  # Framingham Heart Study
   vdb-dump phs000424 --output-path ./data/methylation/framingham
   ```

#### Option B: Demo Mode

For testing, the notebook can generate synthetic methylation data.

### 3. Training

#### Interactive Development (Jupyter)

```bash
jupyter lab methylation_foundation_model.ipynb
```

Then run the cells sequentially to:
- Load and process methylation data
- Configure model architecture
- Train on selected scale
- Evaluate on benchmarks

#### Production Training (Command Line)

```bash
# Small-scale training (single GPU)
python train_production.py \
    --model NT-500M \
    --scale small \
    --data-dir ./data/methylation \
    --output-dir ./checkpoints

# Medium-scale training (4 GPUs)
accelerate launch --num_processes 4 train_production.py \
    --model NT-500M \
    --scale medium \
    --data-dir ./data/methylation \
    --wandb-project methylation-model

# Large-scale training (8 GPUs, AWS p4d.24xlarge)
accelerate launch --num_processes 8 train_production.py \
    --model NT-500M \
    --scale large \
    --method hybrid \
    --epochs 20 \
    --fp16 \
    --wandb-project methylation-model
```

### 4. Evaluation

```bash
# Evaluate on GUE benchmarks
python train_production.py \
    --eval-only \
    --eval-gue \
    --model NT-500M \
    --checkpoint ./checkpoints/best_model

# Or use the notebook for interactive evaluation
```

## Project Structure

```
methylation-model/
├── methylation_foundation_model.ipynb  # Main notebook
├── train_production.py                 # Production training script
├── requirements.txt                    # Python dependencies
├── README.md                          
├── validate_methylation_data.py       # Data validation utility
│
├── data/
│   └── methylation/                   # Methylation data from dbGaP
│       ├── framingham/
│       ├── goldn/
│       └── mesa/
│
├── checkpoints/                       # Model checkpoints
│   └── NT-500M_large/
│
└── results/                          # Evaluation results
    ├── gue_benchmarks.csv
    ├── methylation_tasks.csv
    └── plots/
```

## Methodology

### Methylation Integration Methods

We implement three approaches for incorporating methylation data:

**Method A: Methylation as Additional Features**
```python
# Add projection layer for methylation values
methylation_embedding = projection(methylation_beta)
combined = sequence_embedding + methylation_embedding
```
- Simple, maintains pre-trained weights
- Limited modeling of sequence-methylation interactions

**Method B: Modified Tokenization**
```python
# Expand vocabulary for methylated bases
vocab += ['mA', 'mC', 'mG', 'mT']
sequence = "ACGT" → "AmCGmT"  # Mark methylated positions
```
- Direct representation of methylation state
- Requires retraining embedding layer

**Method C: Multi-Modal Architecture (Recommended)**
```python
# Separate encoders with cross-attention
seq_features = sequence_encoder(sequence)
meth_features = methylation_encoder(beta_values)
fused = cross_attention(seq_features, meth_features)
```
- Most flexible, models complex interactions
- More parameters, slower training

### Training Pipeline

1. **Data Ingestion**
   - Load Illumina 450K/EPIC array data
   - Apply quality filters (beta ∈ [0,1], coverage ≥ 10x)
   - Harmonize across studies (ComBat for batch effects)

2. **Preprocessing**
   - Extract genomic context (±512bp around CpG sites)
   - Tokenize sequences
   - Create train/val/test splits (80/10/10)

3. **Model Training**
   - Base model: Nucleotide Transformer 500M
   - Fine-tuning: LoRA (r=16, α=32)
   - Optimization: AdamW (lr=5e-5, wd=0.01)
   - Regularization: Dropout 0.1, gradient clipping

4. **Evaluation**
   - GUE benchmarks (28 tasks)
   - Methylation-specific tasks (age, tissue, disease)
   - Statistical analysis (MCC, AUC, MAE)

## Evaluation Metrics

### GUE Benchmark Tasks

| Category | Tasks | Metric |
|----------|-------|--------|
| Promoter | prom_core_all, prom_tata, prom_notata | MCC |
| Histone | emp_H3K4me3, emp_H3K36me3, emp_H3K27me3 | MCC |
| Splice Site | splice_reconstructed, splice_acceptor | MCC |
| TF Binding | tf_binding_* | AUC |

### Methylation-Specific Tasks

1. **Age Prediction** (Epigenetic Clock)
   - Uses 353 Horvath clock CpG sites
   - Metric: Mean Absolute Error (years)
   - Baseline: 4.9 years (Horvath 2013)

2. **Tissue Classification**
   - 5+ tissue types (blood, brain, liver, lung, muscle)
   - Metric: Accuracy, F1 per tissue
   - Baseline: 87% (Moss et al. 2018)

3. **Cancer Detection**
   - Multi-cancer classification
   - Metric: AUC-ROC, sensitivity, specificity
   - Baseline: 0.82 AUC (Capper et al. 2018)

## Cloud Deployment

### AWS Setup

```bash
# 1. Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type p4d.24xlarge \
    --key-name your-key \
    --security-group-ids sg-xxxxx

# 2. SSH into instance
ssh -i your-key.pem ec2-user@instance-ip

# 3. Setup environment
git clone https://github.com/your-repo/methylation-model.git
cd methylation-model
bash setup_aws.sh  # Installs CUDA, conda, dependencies

# 4. Download data from S3
aws s3 sync s3://your-bucket/methylation-data ./data/methylation/

# 5. Launch training
accelerate launch --num_processes 8 train_production.py \
    --scale large \
    --data-dir ./data/methylation \
    --output-dir s3://your-bucket/checkpoints
```

### Scaling Configurations

| Scale | GPUs | Instance | Batch Size | Time | Cost |
|-------|------|----------|------------|------|------|
| Small | 1 | g4dn.xlarge | 8 | 2-4h | $5-10 |
| Medium | 4 | p3.8xlarge | 32 | 4-8h | $50-100 |
| Large | 8 | p4d.24xlarge | 64 | 8-16h | $200-400 |
| XL | 16 | Multi-node | 128 | 1-2d | $1000-2000 |


**Interpretation:**
- **H3K4me3 (histone methylation)** is directly related to DNA methylation patterns
- Methylation-enhanced model captures epigenetic signals that sequence-only models miss
- The promoter task relies more on sequence motifs, where methylation focus may dilute performance

### Recommendations

1. **Implement Hybrid Architecture**
   ```python
   class HybridModel:
       def __init__(self):
           self.seq_pathway = SequenceEncoder()  # For motif tasks
           self.meth_pathway = MethylationEncoder()  # For epigenetic tasks
           self.fusion = AdaptiveFusion()  # Task-specific weighting
   ```
   Expected impact: +10-15% on sequence tasks while maintaining epigenetic performance

2. **Expand Training Data**
   - Add WHI (phs001189): 16,000 samples
   - Add MESA (phs000710): Multi-ethnic diversity
   - Expected impact: +5-10% across all tasks

3. **Complete GUE Benchmark**
   - Evaluate on all 28 tasks (currently tested: 2)
   - Identify specific strengths/weaknesses

**Medium-term:**
- Hyperparameter optimization (LoRA rank, learning rate)
- Task-specific adapter heads
- Data augmentation with synthetic methylation

**Long-term:**
- Pre-training on methylation data from scratch
- Multi-omics integration (ATAC-seq, ChIP-seq, Hi-C)
- Cell-type specific models

## References

**Genomic Foundation Models:**
- Nucleotide Transformer: [Dalla-Torre et al., 2023](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2)
- DNABERT-2: [Zhou et al., 2023](https://arxiv.org/abs/2306.15006)

**Methylation:**
- Horvath Clock: [Horvath, 2013](https://genomebiology.biomedcentral.com/articles/10.1186/gb-2013-14-10-r115)
- CpGPT: [Camillo et al., 2023](https://github.com/lcamillo/CpGPT)

**Benchmarks:**
- GUE: [MAGICS Lab](https://github.com/MAGICS-LAB/DNABERT_2)

