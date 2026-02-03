# Neuro-Fuzzy Smart Home Energy Management System ⚡

> 🚧 **Project Status: Work in Progress**  
> This project is currently under active development and continuous improvement.  
> The repository does **not** represent the final version of the system.

This project implements an intelligent smart home energy management system using:

- LSTM neural networks for energy consumption and solar prediction
- Fuzzy logic for adaptive decision-making
- Streamlit dashboard for visualization and interaction

## Files
- `prediction.py` — LSTM prediction models  
- `decision.py` — Fuzzy logic controller  
- `dashboard.py` — Interactive Streamlit dashboard  

## How to Run

```bash
pip install streamlit tensorflow numpy pandas
streamlit run dashboard.py
