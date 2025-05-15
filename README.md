AnimalCLEF2025
This repository implements three approaches for the AnimalCLEF2025 competition, aimed at identifying animal subspecies (loggerhead turtle, lynx, salamander) using one-shot or few-shot learning. The approaches are:

Siamese Network: ConvNeXt backbone with contrastive loss (CPU-only).
ArcFace + FAISS: ConvNeXt with ArcFace loss and FAISS for similarity search (CPU-only).
Prototypical Networks: Few-shot learning with episodic training (CPU-only).

Evaluation uses BAKS (Balanced Accuracy for Known Species), BAUS (Balanced Accuracy for Unknown Species), and their geometric mean.
Requirements

Python 3.8+
CPU-only environment
Dependencies: requirements.txt

Installation

Clone the repository:
git clone https://github.com/yourusername/AnimalCLEF25.git
cd AnimalCLEF25


Install dependencies:
pip install -r requirements.txt


(Optional) Install as a package:
pip install .



Dataset
The AnimalCLEF2025 dataset is not included in this repository due to licensing restrictions that prohibit redistribution. You must download it and place it in the data/ directory.

Download: Obtain the dataset from the official competition page: https://kaggle.com/competitions/animal-clef-2025.
You may need to register for the competition and accept its rules.


Structure: Place the dataset in data/ with the following structure:data/
├── metadata.csv
├── images/
│   ├── SeaTurtleID2022/
│   │   ├── database/
│   │   └── query/
│   ├── LynxID2025/
│   │   ├── database/
│   │   └── query/
│   ├── SalamanderID2025/
│   │   ├── database/
│   │   └── query/


See data/README.md for detailed instructions.

Usage
Run an approach using main.py with a config file:
python main.py --config configs/siamese_config.yaml
python main.py --config configs/arcface_config.yaml
python main.py --config configs/proto_config.yaml

Optional hyperparameter tuning for ArcFace (Approach 2):
python main.py --config configs/arcface_config.yaml --tune

This will:

Load and preprocess the dataset from data/.
Train the selected model (e.g., Siamese Network for 20 epochs, ArcFace for 20 epochs, Prototypical for 10 epochs) on CPU.
Save the model to submissions/.
Generate a submission file in submissions/.

Structure
AnimalCLEF25/
├── configs/                    # Configuration files
├── data/                       # Dataset (excluded from Git)
├── notebooks/                  # Exploratory notebooks
├── scripts/                    # Utility scripts
├── approach1/                  # Siamese Network
├── approach2/                  # ArcFace + FAISS
├── approach3/                  # Prototypical Networks
├── evaluation/                 # Evaluation metrics
├── submissions/                # Submission CSVs and models
├── main.py                     # Main script

Configuration
Edit YAML files in configs/ to adjust hyperparameters:

siamese_config.yaml: data_root, batch_size (32), num_epochs (20), embedding_dim (128).
arcface_config.yaml: data_root, batch_size (32), num_epochs (20), embedding_size (512), new_threshold (0.6).
proto_config.yaml: data_root, batch_size (64), pretrain_epochs (5), epochs (10), embedding_size (1024).

Notes

The Siamese Network may overpredict LynxID2025_lynx_37. Consider weighted sampling or threshold tuning.
Validation set issues are handled with a fallback in approach1/train.py.
Expected geometric mean: ~0.7-0.8.
Approach 2 includes hyperparameter tuning with Optuna.
Approach 3 includes t-SNE visualization of embeddings.

Citations
This project uses the AnimalCLEF2025 dataset and may leverage methods or data from related works. Please cite the following if you use this code or the datasets:

AnimalCLEF25 Competition:
AnimalCLEF25 @ CVPR-FGVC & LifeCLEF. https://kaggle.com/competitions/animal-clef-2025, 2025. Kaggle.


WildlifeDatasets Toolkit (used for methodology in Approach 3):
@inproceedings{vcermak2024wildlifedatasets,
  title={WildlifeDatasets: An open-source toolkit for animal re-identification},
  author={{\v{C}}erm{\'a}k, Vojt{\v{e}}ch and Picek, Lukas and Adam, Luk{\'a}{\v{s}} and Papafitsoros, Kostas},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5953--5963},
  year={2024}
}


SeaTurtleID2022 (if used in the dataset):
@inproceedings{adam2024seaturtleid2022,
  title={SeaTurtleID2022: A long-span dataset for reliable sea turtle re-identification},
  author={Adam, Luk{\'a}{\v{s}} and {\v{C}}erm{\'a}k, Vojt{\v{e}}ch and Papafitsoros, Kostas and Picek, Lukas},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={7146--7156},
  year={2024}
}


WildlifeReID-10k (if used as a training database):
@article{adam2024wildlifereid,
  title={WildlifeReID-10k: Wildlife re-identification dataset with 10k individual animals},
  author={Adam, Luk{\'a}{\v{s}} and {\v{C}}erm{\'a}k, Vojt{\v{e}}ch and Papafitsoros, Kostas and Picek, Lukas},
  journal={arXiv preprint arXiv:2406.09211},
  year={2024}
}



License
MIT License for the code in this repository. Note that the datasets (AnimalCLEF2025, SeaTurtleID2022, WildlifeReID-10k) have their own licenses that prohibit redistribution and commercial use. You must comply with their terms when using the data.
