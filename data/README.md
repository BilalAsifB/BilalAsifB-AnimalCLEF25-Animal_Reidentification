Dataset Instructions
The AnimalCLEF2025 dataset must be downloaded and placed in this directory (data/). The dataset is not included in the repository due to licensing restrictions that prohibit redistribution.
Setup

Download the AnimalCLEF2025 dataset from the official competition page: https://kaggle.com/competitions/animal-clef-2025.
You may need to register for the competition and accept its rules to access the data.


Extract the dataset and place it in the data/ directory with the following structure:data/
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


Ensure metadata.csv contains columns: path, identity, species, image_id, split.

Notes

The path column in metadata.csv should be relative to the data/ directory (e.g., images/SeaTurtleID2022/database/turtles-data/data/images/t001/image1.jpg).
If the dataset structure differs, update the dataset loading logic in approach1/dataset.py, approach2/dataset.py, or approach3/dataset.py.
The dataset may include data from SeaTurtleID2022 or WildlifeReID-10k. If used, you must comply with their licenses and provide proper citations (see README.md).

