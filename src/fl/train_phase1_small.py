from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from src.models import MoETextClassifier
from src.fl import FedServer, local_train
from src.data import build_ag_news_clients


@dataclass
class FLConfigPhase1:
    seq_len: int = 64
    embed_dim: int = 64
    num_experts: int = 4
    expert_hidden_dim: int = 128
    top_k: int = 2
    lora_r: int = 8

    num_clients: int = 4
    rounds: int = 15
    local_epochs: int = 2
    batch_size: int = 16
    lr: float = 5e-4
    repeat: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=" * 65)
    print("  Phase 1: Baseline FL with MoE + LoRA (small corpus)")
    print("=" * 65)
    set_seed(42)
    cfg = FLConfigPhase1()
    device = torch.device(cfg.device)

    clients, test_dataset, vocab_size, num_classes, _vocab = build_ag_news_clients(
        num_clients=cfg.num_clients, seq_len=cfg.seq_len,
        use_external_csv=False, repeat=cfg.repeat,
    )
    train_eval_dataset: Dataset = ConcatDataset(clients)

    print(f"  Clients: {len(clients)}, "
          f"Train samples: {len(train_eval_dataset)}, "
          f"Test samples: {len(test_dataset)}")
    print(f"  Vocab: {vocab_size}, Classes: {num_classes}")
    print(f"  Rounds: {cfg.rounds}, Local epochs: {cfg.local_epochs}, "
          f"LR: {cfg.lr}, Batch: {cfg.batch_size}")
    print("-" * 65)

    global_model = MoETextClassifier(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        num_classes=num_classes,
        num_experts=cfg.num_experts,
        expert_hidden_dim=cfg.expert_hidden_dim,
        k=cfg.top_k,
        lora_r=cfg.lora_r,
    )
    server = FedServer(global_model, device=device)

    for rnd in range(1, cfg.rounds + 1):
        client_states = []
        for client_dataset in clients:
            client_model = MoETextClassifier(
                vocab_size=vocab_size,
                embed_dim=cfg.embed_dim,
                num_classes=num_classes,
                num_experts=cfg.num_experts,
                expert_hidden_dim=cfg.expert_hidden_dim,
                k=cfg.top_k,
                lora_r=cfg.lora_r,
            )
            client_model.load_state_dict(server.get_global_state(), strict=False)

            full_state, _, n, _, _, _, _ = local_train(
                client_model,
                client_dataset,
                epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                device=device,
            )
            client_states.append((full_state, n))

        server.aggregate(client_states)
        train_acc = evaluate(server.global_model, train_eval_dataset, cfg.batch_size, device)
        test_acc = evaluate(server.global_model, test_dataset, cfg.batch_size, device)
        print(f"  Round {rnd:2d}:  train_acc = {train_acc:.4f}  |  test_acc = {test_acc:.4f}")

    print("-" * 65)
    print("  Phase 1 complete.")


if __name__ == "__main__":
    main()

