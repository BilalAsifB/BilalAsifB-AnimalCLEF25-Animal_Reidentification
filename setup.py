from setuptools import setup, find_packages

setup(
    name="AnimalCLEF25",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit-learn",
        "faiss-cpu",
        "optuna",
        "seaborn",
        "wildlife-datasets",
        "matplotlib",
        "Pillow",
        "tqdm",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AnimalCLEF2025: One-shot learning for animal re-identification",
    license="MIT",
    url="https://github.com/yourusername/AnimalCLEF25",
)