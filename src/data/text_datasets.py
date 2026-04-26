from typing import List, Tuple, Union
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split


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
    max_vocab: int = 5000,
) -> Tuple[List[Dataset], Dataset, int, int, dict]:
    """
    Build a text dataset and split into client datasets.

    If CSV files `data/ag_news_train.csv` and `data/ag_news_test.csv` exist,
    they are used as a larger corpus (label,text). Otherwise a small built-in
    corpus is used so the code still runs offline.

    ``max_vocab`` caps the vocabulary to the N most frequent tokens so
    the embedding layer stays small enough for meaningful sparse-vs-dense
    communication comparisons.
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

        # Build frequency-capped vocab: keep only the most common tokens
        from collections import Counter
        token_counts: Counter = Counter()
        for text in texts_train:
            token_counts.update(text.lower().split())

        vocab = {"<pad>": 0}
        for tok, _ in token_counts.most_common(max_vocab - 1):
            vocab[tok] = len(vocab)

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
    return clients, test_dataset, len(vocab), num_classes, vocab


# ============================================================================
# Non-IID partitioning via Dirichlet distribution
# ============================================================================
# Real federated learning almost never has IID data across clients.  The
# standard simulation method (Yurochkin et al. 2019; Hsu et al. 2019) is to
# draw each class's per-client proportions from a Dirichlet(alpha) distribution:
#
#     alpha -> 0    : extreme non-IID (each client sees only 1 or 2 classes)
#     alpha = 0.5   : realistic non-IID for FL benchmarks
#     alpha = 1.0   : moderate skew
#     alpha -> inf  : approaches IID (all clients see all classes equally)


def _dataset_labels(dataset: Dataset) -> np.ndarray:
    """Extract all labels from a dataset, handling Subset indirection."""
    if isinstance(dataset, Subset):
        # Prefer the underlying indices + labels (avoids loading input_ids)
        base = dataset.dataset
        if hasattr(base, "labels"):
            return np.asarray([base.labels[i] for i in dataset.indices])
        return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


def dirichlet_split(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    min_size: int = 20,
    seed: int = 42,
) -> List[Subset]:
    """
    Split ``dataset`` into ``num_clients`` non-IID shards using a Dirichlet
    distribution on class proportions.

    Args:
        dataset: a labelled Dataset (must expose labels via .labels or __getitem__)
        num_clients: number of client shards to produce
        alpha: Dirichlet concentration.  Small -> very non-IID.
        min_size: re-sample the Dirichlet if any client would have fewer than
                  this many samples (avoids empty/near-empty clients)
        seed: RNG seed for reproducibility

    Returns:
        List of Subsets, one per client.
    """
    rng = np.random.default_rng(seed)
    labels = _dataset_labels(dataset)
    n_classes = int(labels.max()) + 1
    n = len(labels)

    # Group sample indices by class
    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
    for idxs in class_indices:
        rng.shuffle(idxs)

    # Re-sample Dirichlet until no client is tiny
    for attempt in range(20):
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            proportions = rng.dirichlet([alpha] * num_clients)
            # Split class-c indices according to these proportions
            split_points = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
            parts = np.split(class_indices[c], split_points)
            for client_id, part in enumerate(parts):
                client_indices[client_id].extend(part.tolist())
        sizes = [len(ids) for ids in client_indices]
        if min(sizes) >= min_size:
            break

    # Shuffle each client's index list so batches are not class-grouped
    for ids in client_indices:
        rng.shuffle(ids)

    return [Subset(dataset, ids) for ids in client_indices]


def client_class_distribution(dataset: Dataset, num_classes: int) -> np.ndarray:
    """Return an (num_classes,) array of class counts for the given shard."""
    labels = _dataset_labels(dataset)
    counts = np.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        counts[c] = int((labels == c).sum())
    return counts


def build_ag_news_clients_noniid(
    num_clients: int = 5,
    alpha: float = 0.5,
    seq_len: int = 64,
    max_vocab: int = 5000,
    use_external_csv: bool = True,
    seed: int = 42,
    min_per_client: int = 20,
) -> Tuple[List[Dataset], Dataset, int, int, dict]:
    """
    Same return signature as build_ag_news_clients, but partitions the training
    set using Dirichlet(alpha) instead of IID random_split.

    Lower alpha = more heterogeneous data across clients.
    """
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    train_csv = data_dir / "ag_news_train.csv"
    test_csv = data_dir / "ag_news_test.csv"

    if use_external_csv and train_csv.exists() and test_csv.exists():
        texts_train, labels_train, texts_test, labels_test = _load_csv_corpus(train_csv, test_csv)
        from collections import Counter
        token_counts: Counter = Counter()
        for text in texts_train:
            token_counts.update(text.lower().split())
        vocab = {"<pad>": 0}
        for tok, _ in token_counts.most_common(max_vocab - 1):
            vocab[tok] = len(vocab)
        train_dataset = TextClassificationDataset(texts_train, labels_train, vocab, seq_len=seq_len)
        test_dataset = TextClassificationDataset(texts_test, labels_test, vocab, seq_len=seq_len)
        num_classes = max(labels_train) + 1 if labels_train else 4
    else:
        texts, labels, vocab = _build_small_corpus(repeat=50)
        full_dataset = TextClassificationDataset(texts, labels, vocab, seq_len=seq_len)
        n_total = len(full_dataset)
        n_test = max(n_total // 5, 1)
        n_train = n_total - n_test
        train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test])
        num_classes = 4

    clients = dirichlet_split(
        train_dataset, num_clients=num_clients,
        alpha=alpha, min_size=min_per_client, seed=seed,
    )
    return clients, test_dataset, len(vocab), num_classes, vocab


