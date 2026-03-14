from setuptools import setup, find_packages

setup(
    name="facescan_model",
    version="0.1.0",
    description="Clinical-grade rPPG model for multi-metric health biomarker estimation",
    author="kpanuragh",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.2",
        "numpy>=1.24.3",
        "opencv-python>=4.8.0",
        "mediapipe>=0.8.11",
        "fastapi>=0.104.1",
        "pandas>=2.0.3",
    ],
)
