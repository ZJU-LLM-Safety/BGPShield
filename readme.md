## System Architecture

BGPShield consists of two core modules operating in a sequential pipeline:

### Overview

```
routing-anomaly-detection/
├── BGPShield/                   # LLM-based Semantic Encoder
│   ├── iterative_as_embeds.py   # AS embedding generation
│   ├── train.py                 # CDR network training
│   └── CDR.py                   # Lightweight dimensionality reduction model
├── anomaly_detector/            # BGP Anomaly Detector
│   ├── diff_evaluator_routeviews.py    # AR-DTW path difference
│   ├── llm_report_anomaly_routeviews.py # Anomaly detection
│   └── utils.py                 # Utility functions
├── routing_monitor/             # Route monitoring
│   ├── all_route_monitor.py     # Collect route changes from 40+ vantage points 
│   └── llmmonitor.py            # AS description construction
├── post_processor/              # Result aggregation
│   ├── alarm_postprocess_routeviews.py
│   ├── rpki_validation_request.py
│   └── summary_routeviews.py
├── data/                        # Data storage
│   ├── bgpstream/
│   ├── caida_as_org/
│   ├── caida_as_rel/
│   └── routeviews/
└── pipeline.sh                  # One-command execution script
```

### Module 1: LLM-based Semantic Encoder (LSE)

**Purpose**: Generate semantic embeddings that capture AS behavioral semantics and routing policy rationales.

#### 1.1 AS Description Construction

Formulates structured natural language descriptions for each AS by integrating:

- **Stable Attributes**: Organization, country, connectivity degree (providers/peers/customers), customer cone size, prefix counts
- **Business Neighbors**: Enumeration of adjacent ASes with relationship labels and their connectivity statistics

**Key Files**:
- `routing_monitor/llmmonitor.py`: Constructs AS descriptions with segment-wise neighbor information
- Related data sources: CAIDA AS Relationship, AS Organization, ASRank datasets (in `data/` directory)

#### 1.2 Segment-wise Embedding Generation

Transforms AS descriptions into LLM embeddings using a segment-wise aggregation scheme:

**Key Files**:
- `BGPShield/iterative_as_embeds.py`: Generates embeddings with iterative neighbor batching
- Supports multiple LLM models (modify `model_path` in the script)

#### 1.3 Contrastive Dimensionality Reduction (CDR)

Compresses high-dimensional LLM embeddings into compact semantic space:

**Key Files**:
- `BGPShield/train.py`: Trains the dimensionality reduction network
- `BGPShield/CDR.py`: Neural network architecture for reduction

### Module 2: BGP Anomaly Detector (BAD)

**Purpose**: Identify anomalous BGP route changes through semantic path analysis.

#### 2.1 Route Change Detection

Monitors BGP updates and identifies route changes:

**Key Files**:
- `routing_monitor/all_route_monitor.py`: Collect route changes from 40+ vantage points
- `routing_monitor/llmmonitor.py`: Additional monitoring utilities

#### 2.2 Path Difference Scoring (AR-DTW Algorithm)

Computes semantic distance between AS paths using AR-DTW:

**Key Files**:
- `anomaly_detector/diff_evaluator_routeviews.py`: Implements AR-DTW algorithm for path difference computation
- `anomaly_detector/utils.py`: Utility functions including threshold computation

#### 2.3 Anomalous Change Detection

Applies adaptive thresholding to identify anomalies:

**Key Files**:
- `anomaly_detector/llm_report_anomaly_routeviews.py`: Anomaly detection and reporting

#### 2.4 Multi-View Event Aggregation

Consolidates anomalous route changes into distinct events:

**Key Files**:
- `post_processor/alarm_postprocess_routeviews.py`: Post-processes alarms and identifies anomaly properties
- `post_processor/rpki_validation_request.py`: RPKI validation checking
- `post_processor/summary_routeviews.py`: Generates final reports
---


## Quick Start with pipeline.sh

The `pipeline.sh` script automates the entire BGPShield workflow, from data collection to anomaly report generation.

### Basic Usage

```bash
./pipeline.sh --year 2008 --month 2 --day 24 --hour 18 --minute 47 \
              --device 0 --reduce True --type ribs
```

### Complete Parameter Reference

```bash
./pipeline.sh \
    --year YYYY \          # Event year (e.g., 2008)
    --month MM \           # Event month (1-12)
    --day DD \             # Event day (1-31)
    --hour HH \            # Event hour in UTC (0-23, default: 12)
    --minute MM \          # Event minute (0-59, default: 0)
    --device GPU_ID \      # GPU device ID (default: 0)
    --reduce BOOL \        # Enable dimensionality reduction (True/False)
    --type DATA_TYPE \     # Data source type (updates/ribs)
    [--llm BOOL] \         # Use LLM embeddings (default: True)
    [--bert BOOL]          # Use BGE-M3 model (default: False)
```

### Parameter Details

#### Required Parameters

- **`--year`, `--month`, `--day`**: Specify the BGP anomaly event date (UTC timezone)
  - Example: `--year 2008 --month 2 --day 24` for the YouTube hijack incident
  
- **`--device`**: GPU device ID for computation
  - Use `nvidia-smi` to check available GPUs
  - Example: `--device 0` (uses GPU 0)

- **`--reduce`**: Enable/disable dimensionality reduction
  - `True`: Apply CDR to compress embeddings (recommended)
  - `False`: Use raw LLM embeddings

- **`--type`**: BGP data source type
  - `updates`: Use UPDATE files (for recent events, post-2015)
  - `ribs`: Use RIB snapshots (for historical events or sparse data periods)

#### Optional Parameters

- **`--hour`, `--minute`**: Specify exact event time
  - Default: `12:00` UTC if not specified
  - Example: `--hour 18 --minute 47` for 18:47 UTC
  - **Data collection window**: ±12 hours from specified time

- **`--llm`**: Choose detection method
  - `True` (default): Use BGPShield with LLM embeddings
  - `False`: Use BEAM baseline for comparison

- **`--bert`**: Select LLM model (only effective when `--llm True`)
  - `False` (default): Use DeepSeek-R1-Distill-Llama-8B
  - `True`: Use BGE-M3 model

---

## Pipeline Workflow

The `pipeline.sh` script executes the following stages sequentially:

### Stage 1: AS Embedding Generation (LSE Module)
- Downloads CAIDA AS relationship data
- Constructs AS descriptions
- Generates LLM embeddings with segment-wise aggregation

### Stage 2: Dimensionality Reduction (if `--reduce True`)
- Trains CDR network on LLM embeddings
- Compresses embeddings to optimal dimension (default: 16)

### Stage 3: Route Change Detection (BAD Module)
- Downloads RouteViews data (RIBs + UPDATEs)
- Identifies route changes in ±12h window

### Stage 4: Path Difference Computation
- Applies AR-DTW algorithm to compute semantic distances
- Generates path difference scores

### Stage 5: Anomaly Detection & Aggregation
- Applies adaptive thresholding
- Aggregates anomalies into events
- Attributes responsible ASes

### Stage 6: Report Generation
- Validates RPKI status
- Identifies anomaly properties
- Generates HTML/JSON/CSV reports
---


## Example Use Cases

### 1. YouTube Hijack (February 2008)

Historical event requiring RIB data:

```bash
./pipeline.sh --year 2008 --month 2 --day 24 --hour 18 --minute 47 \
              --device 0 --reduce True --type ribs
```

**Note**: Events before 2015 should use `--type ribs` due to limited UPDATE file availability.

### 2. Vodafone Route Leak (April 2021)

Recent event with abundant UPDATE data:

```bash
./pipeline.sh --year 2021 --month 4 --day 16 --hour 12 --minute 0 \
              --device 1 --reduce True --type updates
```

### 3. CelerBridge Hijack (August 2022) - Using BGE-M3

Test with alternative LLM model:

```bash
./pipeline.sh --year 2022 --month 8 --day 17 --hour 12 --minute 0 \
              --device 0 --reduce True --type updates --bert True
```

### 4. Comparison with BEAM Baseline

Run BEAM for performance comparison:

```bash
./pipeline.sh --year 2020 --month 4 --day 1 --hour 12 --minute 0 \
              --device 0 --reduce False --type updates --llm False
```

### 5. Recent Event (2025) - Testing Generalization

Evaluate on unseen events after LLM training cutoff:

```bash
./pipeline.sh --year 2025 --month 11 --day 6 --hour 12 --minute 0 \
              --device 0 --reduce True --type updates
```
---
