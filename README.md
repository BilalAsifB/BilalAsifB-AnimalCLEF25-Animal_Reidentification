\documentclass{article}
\usepackage{geometry}
\usepackage{listings}
\usepackage{enumitem}
\usepackage{hyperref}

\title{AnimalCLEF2025}
\date{}
\begin{document}

\maketitle

\section*{Overview}

This repository implements three approaches for the \textbf{AnimalCLEF2025} competition, aimed at identifying animal subspecies (loggerhead turtle, lynx, salamander) using one-shot or few-shot learning. The approaches are:

\begin{itemize}
  \item \textbf{Siamese Network:} ConvNeXt backbone with contrastive loss (CPU-only).
  \item \textbf{ArcFace + FAISS:} ConvNeXt with ArcFace loss and FAISS for similarity search (CPU-only).
  \item \textbf{Prototypical Networks:} Few-shot learning with episodic training (CPU-only).
\end{itemize}

Evaluation uses \textbf{BAKS} (Balanced Accuracy for Known Species), \textbf{BAUS} (Balanced Accuracy for Unknown Species), and their geometric mean.

\section*{Requirements}
\begin{itemize}
  \item Python 3.8+
  \item CPU-only environment
  \item Dependencies listed in \texttt{requirements.txt}
\end{itemize}

\section*{Installation}

\subsection*{Clone the repository}
\begin{lstlisting}[language=bash]
git clone https://github.com/yourusername/AnimalCLEF25.git
cd AnimalCLEF25
\end{lstlisting}

\subsection*{Install dependencies}
\begin{lstlisting}[language=bash]
pip install -r requirements.txt
\end{lstlisting}

\subsection*{(Optional) Install as a package}
\begin{lstlisting}[language=bash]
pip install .
\end{lstlisting}

\section*{Dataset}

The \textbf{AnimalCLEF2025} dataset is not included in this repository due to licensing restrictions that prohibit redistribution. You must download it and place it in the \texttt{data/} directory.

\subsection*{Download}
Obtain the dataset from the official competition page: \\
\url{https://kaggle.com/competitions/animal-clef-2025}. \\
You may need to register for the competition and accept its rules.

\subsection*{Structure}

Place the dataset in \texttt{data/} with the following structure:

\begin{verbatim}
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
\end{verbatim}

See \texttt{data/README.md} for detailed instructions.

\section*{Usage}

Run an approach using \texttt{main.py} with a config file:
\begin{lstlisting}[language=bash]
python main.py --config configs/siamese_config.yaml
python main.py --config configs/arcface_config.yaml
python main.py --config configs/proto_config.yaml
\end{lstlisting}

Optional hyperparameter tuning for ArcFace (Approach 2):
\begin{lstlisting}[language=bash]
python main.py --config configs/arcface_config.yaml --tune
\end{lstlisting}

This will:
\begin{itemize}
  \item Load and preprocess the dataset from \texttt{data/}.
  \item Train the selected model on CPU (e.g., Siamese Network for 20 epochs).
  \item Save the model to \texttt{submissions/}.
  \item Generate a submission file in \texttt{submissions/}.
\end{itemize}

\section*{Structure}

\begin{verbatim}
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
\end{verbatim}

\section*{Configuration}

Edit YAML files in \texttt{configs/} to adjust hyperparameters:

\begin{itemize}
  \item \texttt{siamese_config.yaml}: \texttt{data\_root}, \texttt{batch\_size} (32), \texttt{num\_epochs} (20), \texttt{embedding\_dim} (128)
  \item \texttt{arcface_config.yaml}: \texttt{data\_root}, \texttt{batch\_size} (32), \texttt{num\_epochs} (20), \texttt{embedding\_size} (512), \texttt{new\_threshold} (0.6)
  \item \texttt{proto_config.yaml}: \texttt{data\_root}, \texttt{batch\_size} (64), \texttt{pretrain\_epochs} (5), \texttt{epochs} (10), \texttt{embedding\_size} (1024)
\end{itemize}

\section*{Notes}

\begin{itemize}
  \item The Siamese Network may overpredict \texttt{LynxID2025\_lynx\_37}. Consider weighted sampling or threshold tuning.
  \item Validation set issues are handled with a fallback in \texttt{approach1/train.py}.
  \item Expected geometric mean: $\sim$0.7--0.8.
  \item Approach 2 includes hyperparameter tuning with Optuna.
  \item Approach 3 includes t-SNE visualization of embeddings.
\end{itemize}

\section*{Citations}

This project uses the AnimalCLEF2025 dataset and may leverage methods or data from related works. Please cite the following if you use this code or the datasets:

\begin{itemize}
  \item \textbf{AnimalCLEF25 Competition:} \\
  AnimalCLEF25 @ CVPR-FGVC \& LifeCLEF. \url{https://kaggle.com/competitions/animal-clef-2025}, 2025. Kaggle.

  \item \textbf{WildlifeDatasets Toolkit:}
  \begin{quote}
  \textit{Vojtěch Čermák, Lukáš Picek, Lukáš Adam, Kostas Papafitsoros.} \\
  WildlifeDatasets: An open-source toolkit for animal re-identification. \\
  Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024.
  \end{quote}

  \item \textbf{SeaTurtleID2022:}
  \begin{quote}
  \textit{Lukáš Adam, Vojtěch Čermák, Kostas Papafitsoros, Lukáš Picek.} \\
  SeaTurtleID2022: A long-span dataset for reliable sea turtle re-identification. \\
  WACV 2024.
  \end{quote}

  \item \textbf{WildlifeReID-10k:}
  \begin{quote}
  \textit{Lukáš Adam, Vojtěch Čermák, Kostas Papafitsoros, Lukáš Picek.} \\
  WildlifeReID-10k: Wildlife re-identification dataset with 10k individual animals. \\
  arXiv preprint arXiv:2406.09211, 2024.
  \end{quote}
\end{itemize}

\section*{License}

MIT License for the code in this repository. Note that the datasets (AnimalCLEF2025, SeaTurtleID2022, WildlifeReID-10k) have their own licenses that prohibit redistribution and commercial use. You must comply with their terms when using the data.

\end{document}
