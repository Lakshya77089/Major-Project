from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from src.models import MoETextClassifier
from src.fl import FedServer, local_train
from src.data import build_ag_news_clients


@dataclass
class FLConfigPhase2:
    seq_len: int = 64
    embed_dim: int = 64
    num_experts: int = 8
    expert_hidden_dim: int = 256
    top_k: int = 2
    lora_r: int = 8

    num_clients: int = 5
    rounds: int = 8
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 2e-3
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


def _model_param_summary(model: nn.Module, top_k: int, num_experts: int) -> None:
    """Print a breakdown of model parameters by component."""
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for n, p in model.named_parameters() if n.startswith("embedding"))
    expert = sum(p.numel() for n, p in model.named_parameters() if "moe.experts" in n)
    other = total - embed - expert
    per_expert = expert // num_experts
    sparse_total = total - (num_experts - top_k) * per_expert
    saving = (1 - sparse_total / total) * 100
    print(f"  Model params:  total={total:,}  "
          f"embedding={embed:,} ({embed/total*100:.0f}%)  "
          f"experts={expert:,} ({expert/total*100:.0f}%)  "
          f"other={other:,} ({other/total*100:.0f}%)")
    print(f"  Dense update: {total:,} params = {total * 4 / 1024:.1f} KB")
    print(f"  Sparse update (Top-{top_k}/{num_experts}): {sparse_total:,} params "
          f"= {sparse_total * 4 / 1024:.1f} KB  ({saving:.1f}% saving)")


def main():
    print("=" * 75)
    print("  Phase 2: AG News — Dense vs Sparse FL with Communication Tracking")
    print("=" * 75)
    set_seed(42)
    cfg = FLConfigPhase2()
    device = torch.device(cfg.device)

    clients, test_dataset, vocab_size, num_classes, _vocab = build_ag_news_clients(
        num_clients=cfg.num_clients, seq_len=cfg.seq_len, use_external_csv=True
    )
    train_eval_dataset: Dataset = ConcatDataset(clients)

    print(f"  Dataset: AG News — {len(train_eval_dataset):,} train, {len(test_dataset):,} test, "
          f"{num_classes} classes")
    print(f"  Clients: {len(clients)}, Rounds: {cfg.rounds}, "
          f"Local epochs: {cfg.local_epochs}")
    print(f"  MoE: {cfg.num_experts} experts, Top-K={cfg.top_k}, "
          f"hidden={cfg.expert_hidden_dim}, LoRA r={cfg.lora_r}")

    # Build model template
    model_kwargs = dict(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        num_classes=num_classes,
        num_experts=cfg.num_experts,
        expert_hidden_dim=cfg.expert_hidden_dim,
        k=cfg.top_k,
        lora_r=cfg.lora_r,
    )

    template_model = MoETextClassifier(**model_kwargs)
    _model_param_summary(template_model, cfg.top_k, cfg.num_experts)

    # Dense and sparse global models start identical
    global_model_dense = MoETextClassifier(**model_kwargs)
    global_model_sparse = MoETextClassifier(**model_kwargs)
    # Sync initial weights
    global_model_sparse.load_state_dict(global_model_dense.state_dict())

    server_dense = FedServer(global_model_dense, device=device)
    server_sparse = FedServer(global_model_sparse, device=device)

    print("-" * 75)
    print(f"  {'Round':>5}  {'Dense Acc':>10} {'Sparse Acc':>11}  "
          f"{'Dense KB':>10} {'Sparse KB':>11} {'Saving':>7}")
    print("-" * 75)

    for rnd in range(1, cfg.rounds + 1):
        dense_states = []
        sparse_states = []
        round_dense_bytes = 0
        round_sparse_bytes = 0

        for client_dataset in clients:
            # Train from dense server's global state
            client_model = MoETextClassifier(**model_kwargs)
            client_model.load_state_dict(server_dense.get_global_state(), strict=False)

            full_state, sparse_state, n, dense_b, sparse_b, top_k_idx, _ = local_train(
                client_model,
                client_dataset,
                epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                device=device,
                top_k_sparse=cfg.top_k,
            )
            round_dense_bytes += dense_b
            round_sparse_bytes += sparse_b
            dense_states.append((full_state, n))
            sparse_states.append((sparse_state, n))

        server_dense.aggregate(dense_states)
        server_sparse.aggregate(sparse_states)

        test_acc_dense = evaluate(server_dense.global_model, test_dataset, cfg.batch_size, device)
        test_acc_sparse = evaluate(server_sparse.global_model, test_dataset, cfg.batch_size, device)

        dense_kb = round_dense_bytes / 1024
        sparse_kb = round_sparse_bytes / 1024
        saving = (1 - sparse_kb / dense_kb) * 100 if dense_kb > 0 else 0

        print(f"  {rnd:5d}  {test_acc_dense:10.4f} {test_acc_sparse:11.4f}  "
              f"{dense_kb:10.1f} {sparse_kb:11.1f} {saving:6.1f}%")

    print("-" * 75)
    print(f"  Communication saving per round: {saving:.1f}% "
          f"(sending Top-{cfg.top_k} of {cfg.num_experts} experts)")
    print("  Phase 2 complete.")


if __name__ == "__main__":
    main()

