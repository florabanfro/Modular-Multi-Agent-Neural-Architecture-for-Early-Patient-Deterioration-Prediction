│
├── PHASE 1: ENVIRONMENT SETUP & DATA INGESTION
│   ├── Step 1.1: Environment configurations and package imports (pandas, torch, etc.)
│   └── Step 1.2: Raw file loading, schema verification, and primary descriptive statistics
│
├── PHASE 2: TARGET LABEL ENGINEERING & COHORT EXTRACTION
│   ├── Step 2.1: Link 'edstays' and 'icustays' to isolate the binary target outcome (Y)
│   └── Step 2.2: Extract the timeline parameters (ED admission vs. ICU transfer offsets)
│
├── PHASE 3: FEATURE PIPELINING & DATA SPLICING
│   ├── Step 3.1: Build Static Features (from 'triage', 'edstays', 'diagnosis')
│   ├── Step 3.2: Structure Sequential Time-Series Tables ('vitalsign', 'pyxis')
│   └── Step 3.3: Implement clinical forward-filling imputation and 1-hour window binning
    └──STEP 3.4: LABS SEQUENCE RESHAPING
│
├── PHASE 4: UNIMODAL EXPERT TRAINING (THE NEURAL AGENTS)
│   ├── Step 4.1: Construct and train the Vitals Agent (Sequential LSTM / TCN)
│   ├── Step 4.2: Construct and train the Static Agent (Tabular MLP)
│   └── Step 4.3: Extract and freeze latent state embeddings from the experts
│
├── PHASE 5: UNCERTAINTY QUANTIFICATION & CENTRAL COORDINATION
│   ├── Step 5.1: Inject Monte Carlo Dropout mechanisms into the expert networks
│   └── Step 5.2: Train the Central Neural Coordinator with uncertainty weighting inputs
│
└── PHASE 6: COMPARATIVE ANALYSIS & METRICS BENCHMARKING
    ├── Step 6.1: Train the Baseline XGBoost model using flattened lag aggregations
    └── Step 6.2: Calculate performance criteria (AUROC, AUPRC, Calibration Curves)