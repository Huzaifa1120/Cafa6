# Protein Function Prediction using ESM-2 Embeddings and Hierarchy-Aware Deep Learning

## Overview

This project implements a machine learning pipeline for **protein function annotation** using the CAFA-6 (Critical Assessment of Functional Annotation) dataset. The system combines state-of-the-art protein language models (ESM-2 650M) with deep learning classifiers and Gene Ontology (GO) hierarchy constraints to predict protein biological functions.

The pipeline predicts Gene Ontology (GO) terms across three primary ontologies:
- **Biological Process (BP)** — what function the protein performs
- **Molecular Function (MF)** — the biochemical activity of the protein
- **Cellular Component (CC)** — where in the cell the protein is located

---

## Project Architecture

### 1. **Data Preparation Phase**

#### 1.1 Sequence Loading
- **Input**: FASTA file containing protein sequences from the CAFA-6 training dataset
- **Process**: 
  - Extracts UniProt accession identifiers from FASTA headers
  - Stores sequences in an indexed dictionary for efficient lookup
- **Output**: Dictionary mapping protein IDs to amino acid sequences

#### 1.2 GO Term Processing
- **Input**: TSV file containing protein-GO term annotations with ontology information
- **Process**:
  - Loads all annotated GO terms for each protein
  - Groups terms by protein ID
  - Filters for proteins with corresponding sequences
- **Output**: DataFrame with protein IDs and their associated GO terms

#### 1.3 Sequence-Annotation Merging
- **Input**: Sequence dictionary and GO term dataframe
- **Process**:
  - Performs inner join on protein IDs
  - Ensures every protein in the final dataset has both sequence and GO annotations
- **Output**: Unified training dataframe with sequences and functional annotations

### 2. **Gene Ontology Graph Processing**

#### 2.1 Graph Construction
- **Input**: Gene Ontology OBO (Open Biomedical Ontology) file
- **Process**:
  - Parses the hierarchical structure of GO terms
  - Constructs a Directed Acyclic Graph (DAG) representation
  - Extracts parent-child relationships (IS_A and PART_OF relations)
- **Output**: NetworkX graph representing the GO hierarchy

#### 2.2 Term Propagation
- **Purpose**: Ensures that if a protein has a specific GO annotation, all parent terms (more general functions) are also included
- **Process**:
  - For each annotated GO term, finds all ancestor terms using graph traversal
  - Adds ancestors to the protein's annotation set
  - Reflects biological reality: specific function implies general function
- **Impact**: Expands annotation set with implicit hierarchy information

#### 2.3 Vocabulary Construction
- **Process**:
  - Collects all unique GO terms (including propagated ancestors)
  - Sorts terms lexicographically for reproducibility
  - Creates bidirectional mapping: GO_ID ↔ Integer Index
- **Output**: Vocabulary of ~39,791 unique GO terms with ontology classification

### 3. **Protein Embedding Generation (ESM-2 650M)**

#### 3.1 ESM-2 Model Selection
- **Model**: ESM-2 t33_650M_UR50D
  - 650M parameters, 33 transformer layers
  - Pre-trained on 250B protein sequences from UniRef50
  - Produces 1280-dimensional embeddings
  - Optimal balance between speed and accuracy

#### 3.2 Embedding Process
- **Input**: Protein sequences (potentially up to several thousand amino acids)
- **Process**:
  - Tokenizes sequences using ESM alphabet
  - Chunks long sequences (>512 residues) to manage GPU memory
  - Runs forward pass through ESM-2 model
  - Extracts representations from final transformer layer (layer 33)
  - Mean-pools over sequence length to produce fixed-size embedding
- **GPU Optimization**:
  - Uses half-precision (float16) for memory efficiency
  - Implements chunked processing for sequences exceeding 512 residues
  - Manages CUDA cache to prevent out-of-memory errors
- **Output**: 1280-dimensional dense vector per protein, saved as PyTorch tensors

#### 3.3 Embedding Storage
- **Format**: PyTorch .pt files containing:
  - `embedding`: 1280-dim float32 tensor
  - `targets`: List of GO term labels for that protein
- **Organization**: Sequential files (sample_0.pt, sample_1.pt, ...) for easy batch loading

### 4. **Supplementary Data Generation**

#### 4.1 Ontology Mapping
- **Purpose**: Track which ontology (BP/MF/CC) each GO term belongs to
- **Process**:
  - Maps each vocabulary index to its ontology namespace
  - Enables evaluation segmented by ontology type
- **Output**: JSON file mapping index → ontology classification

#### 4.2 Hierarchy Relations Extraction
- **Purpose**: Extract biological constraints for loss function
- **Process**:
  - Identifies all parent-child relationships from OBO graph
  - Filters to keep only relations between terms in vocabulary
  - Removes duplicates
- **Output**: Set of ~500K-1M (child_idx, parent_idx) tuples
- **Usage**: Enforces hierarchy constraints during model training

### 5. **Deep Learning Model Training**

#### 5.1 Model Architecture
```
Input (1280-dim ESM Embedding)
    ↓
Linear Layer (1280 → 512)
    ↓
BatchNorm + ReLU + Dropout(0.2)
    ↓
Linear Layer (512 → 256)
    ↓
ReLU
    ↓
Output Layer (256 → 39791)  [Multi-hot GO predictions]
```

**Rationale**: 
- Two hidden layers provide sufficient capacity for complex function prediction
- BatchNorm and Dropout prevent overfitting
- Output size matches vocabulary size for multi-label classification

#### 5.2 Loss Function: Hierarchy-Aware Approach
```
Total Loss = BCE Loss + λ × Hierarchy Constraint Loss
           = BCE(y_pred, y_true) + λ × HierarchyViolation
```

**Components**:
- **BCE Loss**: Binary Cross-Entropy for each GO term independently
- **Hierarchy Constraint**: Penalizes cases where a child term has higher confidence than its parent
  - If child_prob > parent_prob, add penalty of max(0, child_prob - parent_prob)
  - Enforces biological rule: parent term probability ≥ child term probability
- **Trade-off Parameter (λ)**: Default 0.3, balances accuracy vs. hierarchy compliance

#### 5.3 Training Configuration
- **Optimizer**: AdamW with learning rate 1e-3, weight decay 1e-4
- **Learning Rate Schedule**: ReduceLROnPlateau (halve LR if validation loss plateaus)
- **Batch Size**: 64 embeddings per batch
- **Data Split**: 90% training (main dataset), 10% validation
- **Epochs**: 40 with early stopping based on F1-score
- **Device**: GPU (CUDA) with automatic fallback to CPU

#### 5.4 Training Loop
1. Forward pass: Pass embeddings through model
2. Loss computation: BCE + hierarchy constraint
3. Backward pass: Gradient computation
4. Optimization step: Parameter updates
5. Validation: Evaluate on held-out 10% with threshold optimization
6. Model checkpoint: Save if validation F1-score improves

### 6. **Inference & Post-Processing**

#### 6.1 Forward Prediction
- **Input**: Protein embeddings from validation set
- **Process**: 
  - Load best trained model
  - Forward pass through classifier
  - Apply sigmoid to get probabilities ∈ [0, 1]
  - Binarize using optimal threshold (typically 0.3-0.4)
- **Output**: Binary predictions (has function / doesn't have function)

#### 6.2 Hierarchy Propagation
- **Purpose**: Enforce GO hierarchy in predictions post-hoc
- **Process**:
  - For each child term predicted: `parent_prob = max(parent_prob, child_prob)`
  - Ensures if specific function is predicted, general category is also predicted
  - Improves biological validity of predictions
- **Impact**: Typically improves recall while maintaining precision

#### 6.3 Evaluation Metrics
**Micro-Averaged Metrics** (recommended for imbalanced multi-label):
- **Precision**: TP / (TP + FP) — of all predicted terms, how many are correct
- **Recall**: TP / (TP + FN) — of all actual terms, how many were found
- **F1-Score**: 2 × (P × R) / (P + R) — harmonic mean balancing precision and recall

**Stratification by Ontology**:
- Global (all terms)
- Biological Process (BP) subset
- Molecular Function (MF) subset
- Cellular Component (CC) subset

---

## Key Technical Features

### Multi-Label Classification
- Each protein can have **multiple** GO annotations (not mutually exclusive)
- Uses multi-hot encoding: binary vector where position i = 1 if GO term i is annotated
- Handles class imbalance through Information Accretion (IA) weighting

### Hierarchy-Aware Learning
- Leverages Gene Ontology's structured knowledge
- Enforces constraints: specific function implies general function
- Custom loss function biases model toward biologically valid predictions
- Post-inference propagation ensures consistency

### Scalability Optimizations
- **Embedding Pre-computation**: ESM-2 model run once, embeddings reused
- **Chunked Sequence Processing**: Handles proteins of any length without OOM
- **Efficient PyTorch Storage**: .pt files preserve tensor format for fast loading
- **Memory Management**: Half-precision embeddings, explicit GPU cache clearing

### Biological Validation
- IA-weighting accounts for term specificity (rare terms weighted higher)
- Micro-averaging treats all sample-term pairs equally
- Separate evaluation per ontology reveals model strengths/weaknesses

---

## File Structure & Dependencies

### Core Dependencies
```
- BioPython (SeqIO): FASTA/TSV file parsing
- Pandas: Data manipulation and aggregation
- OBOnet & NetworkX: GO graph processing and traversal
- PyTorch: Deep learning framework
- fair-esm: ESM-2 protein language model
- Scikit-learn: F1-score and utility metrics
- NumPy: Numerical operations
- tqdm: Progress bar visualization
```

### Generated Artifacts
```
esm_embeddings_t33_650M/
├── sample_0.pt
├── sample_1.pt
├── ...
└── sample_N.pt          # Pre-computed 1280-dim protein embeddings

go_ontology_map_650M.json     # Index → Ontology (BP/MF/CC) mapping
best_model_hierarchy.pth      # Trained model weights (checkpoint)
esm_embeddings_650M.zip       # Compressed embeddings archive
```

---

## Workflow Summary

```
1. LOAD DATA
   ├─ train_sequences.fasta → Protein sequences
   ├─ train_terms.tsv → GO annotations
   └─ go-basic.obo → Gene Ontology hierarchy

2. PROCESS & EXPAND
   ├─ Extract sequence-annotation pairs
   ├─ Propagate GO terms through hierarchy
   └─ Build vocabulary & GO graph

3. GENERATE EMBEDDINGS (ESM-2 650M)
   ├─ Tokenize & chunk sequences
   ├─ Run ESM-2 forward pass
   ├─ Mean-pool representations
   └─ Save .pt files

4. TRAIN DEEP LEARNING MODEL
   ├─ Load embeddings + targets
   ├─ Initialize MLP classifier
   ├─ Run 40 epochs with hierarchy-aware loss
   └─ Save best model checkpoint

5. EVALUATE ON VALIDATION SET
   ├─ Run inference (forward pass)
   ├─ Optimize prediction threshold
   ├─ Apply hierarchy propagation
   └─ Calculate micro-averaged metrics
```

---

## Expected Results

### Performance Metrics (Typical)
| Ontology | Micro F1 | Precision | Recall | Threshold |
|----------|----------|-----------|--------|-----------|
| Global   | ~0.45-0.55 | ~0.50-0.60 | ~0.40-0.50 | 0.30-0.40 |
| BP       | ~0.48-0.58 | ~0.55-0.65 | ~0.42-0.52 | 0.30-0.35 |
| MF       | ~0.42-0.52 | ~0.48-0.58 | ~0.38-0.48 | 0.35-0.45 |
| CC       | ~0.50-0.60 | ~0.55-0.65 | ~0.45-0.55 | 0.25-0.35 |

*Note: Actual results depend on data quality, training duration, and hyperparameter tuning.*

---

## Usage Instructions

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- Kaggle API credentials (for CAFA-6 dataset download)

### Execution Order
```bash
# 1. Install dependencies
!pip install biopython pandas obonet networkx fair-esm torch scikit-learn

# 2. Run cells sequentially in the notebook:
#    - Cells 1-7: Load and prepare data
#    - Cells 8-9: Load GO graph and prepare vocabulary
#    - Cell 10: Generate ESM-2 embeddings (time-intensive)
#    - Cell 11: Compress embeddings
#    - Cells 12-19: Prepare supplementary data
#    - Cell 20: Train model with hierarchy-aware loss
#    - Cells 21-26: Evaluate and report results
```

### Key Hyperparameters (Tunable)
- **ESM Model**: Change `esm2_t33_650M_UR50D` to `esm2_t12_35M_UR50D` for speed
- **Batch Size**: Increase to 128-256 for larger GPU VRAM
- **Max Sequence Length**: Default 1024; reduce for memory constraints
- **Hierarchy Weight (λ)**: Range 0.1-0.5; higher = stricter hierarchy enforcement
- **Prediction Threshold**: Range 0.1-0.9; lower = higher recall, higher = higher precision

---

## Biological Significance

### Why Gene Ontology Hierarchy Matters
The GO forms a Directed Acyclic Graph (DAG) where:
- **Leaf nodes** = specific functions (e.g., "serine protease activity")
- **Root nodes** = general categories (e.g., "molecular_function")
- **Constraint**: If a protein performs a specific function, it implicitly performs all parent (more general) functions

### Advantages of Hierarchy-Aware Learning
1. **Biological Validity**: Predictions respect functional hierarchy
2. **Improved Recall**: Catching parent terms even if exact child isn't predicted
3. **Reduced Spurious Predictions**: Hierarchy constraints act as regularization
4. **Information Integration**: Leverages expert-curated knowledge structure

---

## Citation & Acknowledgments

- **ESM-2 Model**: Linearity-scaling Transformer Learned with Masked Sequence Modeling (Meta AI, 2022)
- **CAFA-6 Dataset**: Critical Assessment of Functional Annotation (Available on Kaggle)
- **Gene Ontology**: Collaborative effort for standardized protein function annotation

---

## License & Author

This implementation is designed for the CAFA-6 protein function prediction challenge.

**Date**: 2024-2025
**Framework**: PyTorch
**Status**: Production-Ready

---

## Troubleshooting

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA Out-of-Memory (ESM) | Batch size too large | Reduce BATCH_SIZE or use smaller ESM model |
| Model not loading | Wrong checkpoint path | Verify path in `model_path` variable |
| Low F1-Score (~0.2) | Random initialization | Ensure best_model weights loaded correctly |
| Slow evaluation | No GPU available | Use CPU with patience or verify CUDA availability |
| Missing GO terms | Vocabulary mismatch | Rebuild GO_ID_TO_INDEX before inference |

---

## Future Enhancements

1. **Multi-Modal Learning**: Incorporate sequence alignment features alongside embeddings
2. **Ensemble Methods**: Combine multiple ESM models for robust predictions
3. **Active Learning**: Iteratively select uncertain predictions for manual annotation
4. **Transfer Learning**: Fine-tune ESM-2 on CAFA-6 specific data
5. **Explainability**: Attribution methods to identify key sequence regions for predictions

---

## Contact & Support

For questions or issues with this implementation, refer to the code comments and docstrings embedded throughout the notebook cells.
