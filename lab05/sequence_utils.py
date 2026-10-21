"""Preprocessing, PyTorch API wiring and experiment execution for LAB05."""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

PAD, UNK = 0, 1


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_vocabulary(token_lists, max_size=20000, min_count=2):
    counts = Counter(token for tokens in token_lists for token in tokens)
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    for token, count in ordered:
        if count >= min_count and len(vocab) < max_size:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocab, max_length):
    ids = [vocab.get(token, UNK) for token in tokens[:max_length]]
    return ids or [UNK]


class ReviewDataset(Dataset):
    def __init__(self, token_lists, labels, vocab, max_length):
        self.ids = [encode(tokens, vocab, max_length) for tokens in token_lists]
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.labels[index]


def collate_reviews(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    x = torch.zeros((len(batch), int(lengths.max())), dtype=torch.long)
    for index, seq in enumerate(sequences):
        x[index, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return x, lengths, torch.tensor(labels, dtype=torch.float32)


def make_loader(frame, vocab, max_length, batch_size, shuffle=False, seed=42):
    dataset = ReviewDataset(frame.tokens.tolist(), frame.label.to_numpy(), vocab, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=collate_reviews,
                      generator=torch.Generator().manual_seed(seed))


class SequenceClassifier(nn.Module):
    def __init__(self, architecture, vocab_size, embedding_dim, hidden_size,
                 seed=42, **_):
        super().__init__()
        self.architecture = architecture
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        with torch.no_grad():
            initial = torch.randn(vocab_size, embedding_dim,
                                  generator=torch.Generator().manual_seed(seed + 1)) * 0.1
            initial[PAD].zero_()
            self.embedding.weight.copy_(initial)
        recurrent_class = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[architecture]
        self.recurrent = recurrent_class(embedding_dim, hidden_size, num_layers=1,
                                         batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, ids, lengths, return_hidden=False):
        embedded = self.embedding(ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                       enforce_sorted=False)
        _, state = self.recurrent(packed)
        hidden = state[0] if self.architecture == "LSTM" else state
        last = hidden[-1]
        logits = self.classifier(last).squeeze(-1)
        return (logits, last) if return_hidden else logits


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, loader, device):
    model.eval()
    total_loss, total = 0.0, 0
    all_probabilities, all_labels = [], []
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.inference_mode():
        for ids, lengths, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids, lengths)
            total_loss += criterion(logits, labels).item()
            total += len(labels)
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    probabilities = np.concatenate(all_probabilities)
    labels = np.concatenate(all_labels).astype(int)
    return total_loss / total, probabilities, labels


def metric_record(labels, probabilities):
    predicted = (np.asarray(probabilities) >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(labels, predicted, average="binary",
                                                pos_label=1, zero_division=0)
    _, _, macro, _ = precision_recall_fscore_support(labels, predicted, average="macro",
                                                   zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(labels, predicted), precision=p, recall=r,
                f1=f, f1_macro=macro, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def train_run(architecture, max_length, train, validation, vocab, config, device, out_dir):
    name = f"{architecture.lower()}_{max_length}"
    seed_everything(config["seed"])
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = SequenceClassifier(architecture, len(vocab), config["embedding_dim"],
                               config["hidden_size"], config["seed"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    train_loader = make_loader(train, vocab, max_length, config["batch_size"],
                                shuffle=True, seed=config["seed"])
    validation_loader = make_loader(validation, vocab, max_length, config["batch_size"])
    history, best_loss, best_state, best_epoch = [], float("inf"), None, None
    training_seconds, validation_seconds, peak_rss = 0.0, 0.0, psutil.Process().memory_info().rss
    synchronize(device)
    wall_start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        synchronize(device)
        start = time.perf_counter()
        total_loss, total_samples, preclip_norm_sum, batches = 0.0, 0, 0.0, 0
        for ids, lengths, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids, lengths), labels)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            preclip_norm_sum += float(norm)
            batches += 1
        synchronize(device)
        epoch_train_seconds = time.perf_counter() - start
        training_seconds += epoch_train_seconds
        start = time.perf_counter()
        validation_loss, probabilities, labels = evaluate(model, validation_loader, device)
        synchronize(device)
        epoch_validation_seconds = time.perf_counter() - start
        validation_seconds += epoch_validation_seconds
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        history.append(dict(model=name, epoch=epoch, train_loss=total_loss / total_samples,
                            validation_loss=validation_loss,
                            validation_f1=metric_record(labels, probabilities)["f1"],
                            mean_preclip_gradient_norm=preclip_norm_sum / batches,
                            training_seconds=epoch_train_seconds,
                            validation_seconds=epoch_validation_seconds))
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"{name}: epoch {epoch}/{config['epochs']} train={history[-1]['train_loss']:.4f} "
              f"val={validation_loss:.4f} val_f1={history[-1]['validation_f1']:.4f} "
              f"train_s={epoch_train_seconds:.1f}", flush=True)
    synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    checkpoint_path = Path(out_dir) / "checkpoints" / f"{name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best_state, architecture=architecture,
                    max_length=max_length, vocab_size=len(vocab), config=config,
                    best_epoch=best_epoch), checkpoint_path)
    model.load_state_dict(best_state)
    loss, probabilities, labels = evaluate(model, validation_loader, device)
    record = dict(model=name, architecture=architecture, representation="Embedding",
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  training_time_seconds=training_seconds, validation_time_seconds=validation_seconds,
                  fit_wall_seconds=wall_seconds, epochs=config["epochs"],
                  selected_epoch=best_epoch, max_length=max_length, vocabulary_size=len(vocab),
                  embedding_dim=config["embedding_dim"], hidden_size=config["hidden_size"],
                  batch_size=config["batch_size"], optimizer="Adam",
                  learning_rate=config["learning_rate"], seed=config["seed"],
                  validation_loss=loss, validation_f1=metric_record(labels, probabilities)["f1"],
                  process_rss_sampled_peak_mb=peak_rss / 1024**2,
                  cuda_peak_allocated_mb=(torch.cuda.max_memory_allocated(device)/1024**2
                                          if device.type == "cuda" else None),
                  checkpoint=str(checkpoint_path.relative_to(Path.cwd())))
    record.update({f"validation_{key}": value for key, value in metric_record(labels, probabilities).items()})
    del optimizer, model, train_loader, validation_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pd.DataFrame(history).to_csv(Path(out_dir)/f"history_{name}.csv", index=False)
    (Path(out_dir)/f"config_{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record, history


def load_run(record, device):
    checkpoint = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
    model = SequenceClassifier(checkpoint["architecture"], checkpoint["vocab_size"],
                               **checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def predict_token_lists(model, token_lists, vocab, max_length, device, batch_size=128):
    dataset = ReviewDataset(token_lists, np.zeros(len(token_lists)), vocab, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_reviews, num_workers=0)
    return evaluate(model, loader, device)[1]


def retained_table(frames, lengths=(30, 50, 100, 200)):
    rows = []
    for split, frame in frames.items():
        original = frame.tokens.map(len).to_numpy()
        for length in lengths:
            retained = np.minimum(original, length)
            rows.append(dict(split=split, max_length=length, documents=len(frame),
                             total_tokens=int(original.sum()), retained_tokens=int(retained.sum()),
                             percent_tokens_retained=100*retained.sum()/original.sum(),
                             percent_documents_truncated=100*np.mean(original > length),
                             percent_documents_padded=100*np.mean(original < length),
                             percent_padding_at_fixed_length=100*(1-retained.sum()/(len(frame)*length))))
    return pd.DataFrame(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()
