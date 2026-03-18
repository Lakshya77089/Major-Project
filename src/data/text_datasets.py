from typing import List, Tuple
import csv
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split


class TextClassificationDataset(Dataset):
    """
    Simple text dataset with a tiny real-ish corpus.

    This is independent of torchtext to avoid binary issues on Windows.
    """

    def __init__(self, texts, labels, vocab, seq_len: int = 64):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def encode(self, text: str):
        tokens = text.lower().split()
        ids = [self.vocab.get(token, 0) for token in tokens]
        if not ids:
            ids = [0]
        ids = ids[: self.seq_len]
        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = self.encode(text)
        return input_ids, torch.tensor(label, dtype=torch.long)


def _build_small_corpus(repeat: int = 20) -> Tuple[List[str], List[int], dict]:
    """
    Build a small 'real' corpus with four topics:
    0 = world, 1 = sports, 2 = business, 3 = tech.
    """
    data = [
        (0, "global leaders meet to discuss climate change and international policy"),
        (0, "earthquake hits coastal city causing severe damage and rescue operations"),
        (0, "election results spark protests in several major countries"),
        (1, "local football team wins championship after dramatic final match"),
        (1, "star striker scores hat trick to secure victory in league game"),
        (1, "olympic committee announces new rules for track and field events"),
        (2, "stock markets rally after central bank cuts interest rates"),
        (2, "startup secures funding to expand its online retail platform"),
        (2, "oil prices fall as global demand slows and supply increases"),
        (3, "tech company unveils new smartphone with advanced ai camera features"),
        (3, "researchers develop efficient neural network model for edge devices"),
        (3, "cybersecurity experts warn about rise in ransomware attacks"),
    ]
    texts = []
    labels = []
    for _ in range(repeat):
        for y, t in data:
            texts.append(t)
            labels.append(y)

    vocab = {"<pad>": 0}
    idx = 1
    for text in texts:
        for tok in text.lower().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return texts, labels, vocab


def _load_csv_corpus(train_csv: Path, test_csv: Path) -> Tuple[List[str], List[int], List[str], List[int]]:
    texts_train, labels_train = [], []
    texts_test, labels_test = [], []

    if train_csv.exists() and test_csv.exists():
        with train_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    label = int(row[0])
                except ValueError:
                    # Skip header or malformed rows
                    continue
                # Map AG_NEWS labels 1..4 -> 0..3
                label = label - 1
                text = row[1]
                labels_train.append(label)
                texts_train.append(text)
        with test_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    label = int(row[0])
                except ValueError:
                    continue
                label = label - 1
                text = row[1]
                labels_test.append(label)
                texts_test.append(text)

    return texts_train, labels_train, texts_test, labels_test


def build_ag_news_clients(
    num_clients: int = 4,
    seq_len: int = 32,
    min_per_client: int = 20,
    repeat: int = 20,
    use_external_csv: bool = True,
) -> Tuple[List[Dataset], Dataset, int, int]:
    """
    Build a text dataset and split into client datasets.

    If CSV files `data/ag_news_train.csv` and `data/ag_news_test.csv` exist,
    they are used as a larger corpus (label,text). Otherwise a small built-in
    corpus is used so the code still runs offline.
    """
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    train_csv = data_dir / "ag_news_train.csv"
    test_csv = data_dir / "ag_news_test.csv"

    texts_train: List[str]
    labels_train: List[int]
    texts_test: List[str]
    labels_test: List[int]

    if use_external_csv and train_csv.exists() and test_csv.exists():
        texts_train, labels_train, texts_test, labels_test = _load_csv_corpus(train_csv, test_csv)
        # Simple vocab from training texts
        vocab = {"<pad>": 0}
        idx = 1
        for text in texts_train:
            for tok in text.lower().split():
                if tok not in vocab:
                    vocab[tok] = idx
                    idx += 1

        train_dataset = TextClassificationDataset(texts_train, labels_train, vocab, seq_len=seq_len)
        test_dataset = TextClassificationDataset(texts_test, labels_test, vocab, seq_len=seq_len)
    else:
        texts, labels, vocab = _build_small_corpus(repeat=repeat)
        full_dataset = TextClassificationDataset(texts, labels, vocab, seq_len=seq_len)

        # Train/test split for backtesting
        n_total = len(full_dataset)
        n_test = max(n_total // 5, 1)
        n_train = n_total - n_test
        train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test])

    lengths = [len(train_dataset) // num_clients] * num_clients
    lengths[0] += len(train_dataset) - sum(lengths)
    clients = list(random_split(train_dataset, lengths))

    clients = [c for c in clients if len(c) >= min_per_client]

    num_classes = max(labels_train or [0]) + 1 if use_external_csv and train_csv.exists() else 4
    return clients, test_dataset, len(getattr(train_dataset, "vocab", vocab)), num_classes


