# Neuro-Fuzzy Smart Home Energy Management System ⚡

> 🚧 **Project Status: Work in Progress**  
> This project is currently under active development and continuous improvement.  
> This repository does not represent the final version of the system.

## Summary

This project implements an intelligent smart home energy management system that combines
LSTM neural networks for energy consumption and solar power prediction with a fuzzy logic
controller for adaptive and explainable decision-making. An interactive Streamlit dashboard
is used to visualize predictions and system decisions in real time.

## Files

- `prediction.py` — LSTM prediction models  
- `decision.py` — Fuzzy logic controller  
- `dashboard.py` — Interactive Streamlit dashboard  

## How to Run

```bash
pip install streamlit tensorflow numpy pandas
streamlit run dashboard.py
