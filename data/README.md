## Dataset Instructions

The **AnimalCLEF2025** dataset must be downloaded and placed in the `data/` directory.

> ⚠️ The dataset is **not included** in this repository due to licensing restrictions that prohibit redistribution.

---

### Setup

1. **Download the dataset**
   Visit the official competition page:
   🔗 [https://kaggle.com/competitions/animal-clef-2025](https://kaggle.com/competitions/animal-clef-2025)

   > You may need to register for the competition and accept its rules to access the data.

2. **Extract and organize the dataset**
   Place the extracted contents into the `data/` directory with the following structure:

```
data/
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
```

3. **Verify metadata**
   The `metadata.csv` file should contain the following columns:

* `path`
* `identity`
* `species`
* `image_id`
* `split`

---

### Notes

* The `path` column in `metadata.csv` should be **relative to the `data/` directory**.
  Example:

  ```
  images/SeaTurtleID2022/database/turtles-data/data/images/t001/image1.jpg
  ```

* If your dataset structure differs from the expected layout, you will need to update the dataset loading logic in one or more of the following:

  * `approach1/dataset.py`
  * `approach2/dataset.py`
  * `approach3/dataset.py`

* The dataset may include external sources such as **SeaTurtleID2022** or **WildlifeReID-10k**.
  If you use these, you must **comply with their licenses** and **cite the sources properly** (see [README.md](../README.md)).

