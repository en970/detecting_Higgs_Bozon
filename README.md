# Higgs Boson Diphoton (H → γγ) Analysis

## About

This project analyzes the Higgs boson decay into two photons using ATLAS open data.

## Physics Background

- **Higgs Boson:** The particle that gives mass to other particles
- **Diphoton Channel:** Higgs decaying into two photons (H → γγ)
- **Invariant Mass:** We expect a "bump" around ~125 GeV


## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Simulation Mode (No Internet Required)
```bash
python src/simulation_analysis.py
```

### Basic Analysis (Requires Internet)
```bash
python src/basic_analysis.py
```

### Advanced Analysis (Requires Internet)
```bash
python src/advanced_analysis.py
```

## Data Source

- **ATLAS Open Data:** https://atlas-opendata.web.cern.ch/
- **Dataset:** 13 TeV Diphoton (GamGam)
