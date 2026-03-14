# Clinical-Grade rPPG Model for Multi-Metric Health Estimation

Shared-feature multi-task learning model for estimating heart rate, respiratory rate, and oxygen saturation from facial video using remote photoplethysmography (rPPG).

## Features

- Multi-metric estimation: HR, RR, SpO2 from single video
- Confidence scores for each metric
- Trained on 3600+ videos from 600+ subjects (MCD-rPPG)
- Clinical-grade accuracy: MAE < 3 bpm (HR), < 2 breaths/min (RR)
- Open-source methodology for medical research
- Web-based real-time inference

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## Quick Start

See `notebooks/` for detailed usage examples.

## Paper & Citation

[Citation TBD after publication]

## License

MIT
