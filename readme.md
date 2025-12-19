________________________________________
# CASPER V5 — Synthetic Recon Governance Console

CASPER is a deterministic, human-gated **synthetic reconnaissance governance system**.  
It fuses ISR-style telemetry, confidence modeling, and audit continuity to maintain a truthful operational picture while explicitly refusing action, targeting, or weaponization.

All data is synthetic.  
CASPER is designed to demonstrate **how recon systems should communicate uncertainty, limits, and human authority**.

---

## Core Principles

- **Recon-Only**  
  No weapon control, no targeting logic, no kinetic recommendations.

- **Deterministic & Reproducible**  
  Every run is seeded and replayable via visible Run ID and RNG seed.

- **Human-Gated Governance**  
  The system never acts autonomously. All consequential changes require an explicit human gate.

- **Audit-Bound**  
  State transitions, proposals, and decisions are hash-chained for post-run inspection.

- **Uncertainty-Forward**  
  Confidence degradation is surfaced visually and behaviorally. The system withdraws rather than over-asserts.

---

## What CASPER Is (and Is Not)

**CASPER is:**
- A synthetic ISR / recon visualization
- A governance and oversight demo
- A confidence and risk communication system

**CASPER is not:**
- A weapon system
- A targeting system
- A flight controller
- An autonomous decision-maker

---

## Features

- Synthetic high-speed UAV telemetry (non-operational)
- Confidence-weighted risk and predicted risk metrics
- Synthetic IR / terrain recon visualization with watermark
- Human-gated proposal and corridor governance logic
- Deterministic run IDs and RNG seeds
- Audit log with hash chaining
- Streamlit single-file architecture

---

## Requirements

### Python
- Python **3.10+** recommended

### Python Dependencies
Install via pip:

```bash
pip install streamlit numpy pandas pydeck matplotlib
Exact packages used:
•	streamlit
•	numpy
•	pandas
•	pydeck
•	matplotlib
________________________________________
Running CASPER
From the project directory:
streamlit run app.py
Then open the local URL provided by Streamlit (usually http://localhost:8501).
________________________________________
Determinism & Reproducibility
Each session generates:
•	Run ID
•	RNG Seed
These are displayed in the UI header.
Using the same seed produces the same synthetic run behavior, enabling replay, debugging, and audit review.
________________________________________
Synthetic Data Disclaimer
SYNTHETIC — NOT OPERATIONAL
All telemetry, imagery, and scenarios are procedurally generated and non-representative of real operations.
CASPER is intended solely for demonstration, research, and governance discussion.
________________________________________
License & Use
MIT
________________________________________

