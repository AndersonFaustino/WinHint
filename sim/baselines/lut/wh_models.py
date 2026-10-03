"""Classifiers for the B5 learned predictor (features -> config).

Library module (no CLI). All models share the interface ``fit(X, y) -> self``
and ``predict(X) -> int array`` and are picklable, so train_phase_classifier.py and
export_lookup_table.py can exchange them through one ``model.pkl``.

    mlp   small multilayer perceptron (PyTorch; 4 -> 32 -> 32 -> n_configs).
          Replaces the old sliding-window Transformer, which could not be
          exported faithfully to a 4-D per-window LUT.
    transformer  tiny Transformer encoder (PyTorch) in the spirit of the original
          B5 classifier, but per window: each of the 4 features is one token
          (value embedding + learned feature embedding), 1 encoder layer,
          mean-pooled, linear head. Being a function of the current window only,
          it exports exactly to the 4-D LUT. Mini-batch training.
    tree  CART decision tree (Gini), pure numpy.
    knn   k-nearest neighbours on standardised features, pure numpy.
    bins  LUT-native model: majority oracle label per LUT cell, empty cells
          filled from the nearest populated cell. Its edges become the LUT edges.

Features are signed-log1p compressed (every feature) and standardised inside
each model (``_Scaler``), so callers pass raw per-window features.
"""

from __future__ import annotations

import numpy as np

#: Model type keys accepted by ``make_model`` (and ``--model``).
MODEL_TYPES = ("mlp", "tree", "knn", "bins", "transformer")
#: PyTorch intra-op threads (small full-batch MLP; the host is shared)
TORCH_THREADS = 2


class _Scaler:
    """Signed-log1p compression followed by per-feature standardisation.

    Attributes:
        mu: Per-feature mean of the compressed training features (set by ``fit``).
        sd: Per-feature standard deviation; values below 1e-9 are replaced by 1.
    """
    def fit(self, X: np.ndarray) -> "_Scaler":
        """Learn the mean and standard deviation of the compressed features.

        Args:
            X: ``(n, n_features)`` raw features.

        Returns:
            ``self``.
        """
        Z = self._pre(X)
        self.mu = Z.mean(axis=0)
        self.sd = Z.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        return self

    @staticmethod
    def _pre(X: np.ndarray) -> np.ndarray:
        """Return ``sign(X) * log1p(|X|)`` as a float array."""
        X = np.asarray(X, dtype=float)
        return np.sign(X) * np.log1p(np.abs(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compress and standardise ``X`` with the fitted statistics.

        Args:
            X: ``(n, n_features)`` raw features.

        Returns:
            Standardised features of the same shape.
        """
        return (self._pre(X) - self.mu) / self.sd


def _majority(y: np.ndarray, n_classes: int, w: np.ndarray | None = None) -> int:
    """Return the (optionally weighted) most frequent label; ties -> smallest index.

    Args:
        y: Integer labels.
        n_classes: Minimum number of classes counted.
        w: Optional per-sample weights.

    Returns:
        The majority config index.
    """
    c = np.bincount(y, weights=w, minlength=n_classes)
    return int(np.argmax(c))  # ties -> smallest config index


class KNNModel:
    """k-nearest-neighbours classifier on standardised features (pure numpy).

    Prediction is the majority label (ties -> smallest config) among the ``k``
    nearest training samples by squared Euclidean distance.

    Attributes:
        name: Model type key (``"knn"``).
        k: Number of neighbours.
        n_classes: Number of configs (``max(y) + 1`` when not given).
        scaler: Fitted ``_Scaler``.
        Xt: Standardised training features.
        y: Training labels.
    """
    name = "knn"

    def __init__(self, k: int = 5, n_classes: int | None = None):
        """Create an unfitted model.

        Args:
            k: Number of neighbours.
            n_classes: Number of configs; ``None`` infers ``max(y) + 1`` in ``fit``.
        """
        self.k = k
        self.n_classes = n_classes

    def fit(self, X, y):
        """Store the standardised training set.

        Args:
            X (numpy.ndarray): ``(n, n_features)`` raw features.
            y (numpy.ndarray): Integer labels.

        Returns:
            model (KNNModel): ``self``.
        """
        y = np.asarray(y, dtype=int)
        self.n_classes = self.n_classes or int(y.max()) + 1
        self.scaler = _Scaler().fit(X)
        self.Xt = self.scaler.transform(X)
        self.y = y
        return self

    def predict(self, X, batch: int = 2048):
        """Predict a config per sample.

        Args:
            X (numpy.ndarray): ``(n, n_features)`` raw features.
            batch: Query rows processed per distance-matrix block.

        Returns:
            configs (numpy.ndarray): ``(n,)`` integer config indices.
        """
        Z = self.scaler.transform(X)
        out = np.empty(len(Z), dtype=int)
        k = min(self.k, len(self.y))
        for s in range(0, len(Z), batch):
            d = ((Z[s:s + batch, None, :] - self.Xt[None, :, :]) ** 2).sum(-1)
            idx = np.argpartition(d, k - 1, axis=1)[:, :k]
            for r, row in enumerate(idx):
                out[s + r] = _majority(self.y[row], self.n_classes)
        return out


class TreeModel:
    """CART with Gini impurity on standardised features.

    Nodes are stored in parallel arrays; a leaf has ``feat == -1``.

    Attributes:
        name: Model type key (``"tree"``).
        max_depth: Maximum depth.
        min_leaf: Minimum samples per child.
        n_classes: Number of configs.
        scaler: Fitted ``_Scaler``.
        feat: Split feature per node (-1 for leaves).
        thr: Split threshold per node (standardised units).
        left: Left child index per node (-1 for leaves).
        right: Right child index per node (-1 for leaves).
        label: Majority label per node.
    """
    name = "tree"

    def __init__(self, max_depth: int = 6, min_leaf: int = 5, n_classes: int | None = None):
        """Create an unfitted tree.

        Args:
            max_depth: Maximum depth (root = 0).
            min_leaf: Minimum samples per child of a split.
            n_classes: Number of configs; ``None`` infers ``max(y) + 1`` in ``fit``.
        """
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_classes = n_classes

    def fit(self, X, y):
        """Grow the tree on standardised features.

        Args:
            X (numpy.ndarray): ``(n, n_features)`` raw features.
            y (numpy.ndarray): Integer labels.

        Returns:
            model (TreeModel): ``self``.
        """
        y = np.asarray(y, dtype=int)
        self.n_classes = self.n_classes or int(y.max()) + 1
        self.scaler = _Scaler().fit(X)
        Z = self.scaler.transform(X)
        # node arrays: feature, threshold, left, right, label
        self.feat, self.thr, self.left, self.right, self.label = [], [], [], [], []
        self._grow(Z, y, 0)
        self.feat = np.array(self.feat); self.thr = np.array(self.thr)
        self.left = np.array(self.left); self.right = np.array(self.right)
        self.label = np.array(self.label)
        return self

    def _new(self, label):
        """Append a leaf node with ``label`` and return its index."""
        for a, v in ((self.feat, -1), (self.thr, 0.0), (self.left, -1), (self.right, -1),
                     (self.label, label)):
            a.append(v)
        return len(self.feat) - 1

    def _gini(self, counts):
        """Return the Gini impurity of each row of class ``counts``."""
        n = counts.sum(axis=-1, keepdims=True)
        p = counts / np.maximum(n, 1)
        return 1.0 - (p ** 2).sum(axis=-1)

    def _grow(self, Z, y, depth):
        """Recursively grow the subtree for samples ``(Z, y)``.

        A node becomes a leaf at ``max_depth``, with fewer than ``2 * min_leaf``
        samples, when it is pure, or when no split lowers the size-weighted Gini
        impurity. Splits put ``Z[:, f] <= threshold`` on the left; thresholds are
        midpoints between distinct consecutive values.

        Args:
            Z (numpy.ndarray): Standardised features of the node's samples.
            y (numpy.ndarray): Labels of the node's samples.
            depth (int): Depth of the node.

        Returns:
            node (int): Index of the created node.
        """
        node = self._new(_majority(y, self.n_classes))
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or np.all(y == y[0]):
            return node
        best = (None, None, self._gini(np.bincount(y, minlength=self.n_classes)[None])[0] * len(y))
        onehot = np.eye(self.n_classes)[y]
        for f in range(Z.shape[1]):
            order = np.argsort(Z[:, f], kind="stable")
            zs, cs = Z[order, f], np.cumsum(onehot[order], axis=0)
            tot = cs[-1]
            n = len(y)
            pos = np.arange(self.min_leaf, n - self.min_leaf + 1)
            if pos.size == 0:
                continue
            valid = zs[pos - 1] < zs[np.minimum(pos, n - 1)]
            pos = pos[valid]
            if pos.size == 0:
                continue
            left = cs[pos - 1]
            right = tot - left
            cost = self._gini(left) * pos + self._gini(right) * (n - pos)
            i = int(np.argmin(cost))
            if cost[i] < best[2] - 1e-12:
                p = pos[i]
                best = (f, 0.5 * (zs[p - 1] + zs[p]), cost[i])
        f, t, _ = best
        if f is None:
            return node
        m = Z[:, f] <= t
        self.feat[node] = f
        self.thr[node] = t
        l = self._grow(Z[m], y[m], depth + 1)
        r = self._grow(Z[~m], y[~m], depth + 1)
        self.left[node], self.right[node] = l, r
        return node

    def predict(self, X):
        """Predict a config per sample by walking the tree.

        Args:
            X (numpy.ndarray): ``(n, n_features)`` raw features.

        Returns:
            configs (numpy.ndarray): ``(n,)`` integer config indices.
        """
        Z = self.scaler.transform(X)
        out = np.empty(len(Z), dtype=int)
        for i, z in enumerate(Z):
            n = 0
            while self.feat[n] >= 0:
                n = self.left[n] if z[self.feat[n]] <= self.thr[n] else self.right[n]
            out[i] = self.label[n]
        return out


class BinsModel:
    """Majority label per LUT cell (edges from per-feature quantiles).

    Works on raw (unscaled) features; its ``edges``/``table`` are exported
    verbatim as the runtime LUT.

    Attributes:
        name: Model type key (``"bins"``).
        n_edges: Requested edges per feature.
        n_classes: Number of configs.
        edges: Ascending upper bin edges per feature (set by ``fit``).
        shape: Bins per feature.
        table: Flat row-major config table.
    """
    name = "bins"

    def __init__(self, n_edges: int = 7, n_classes: int | None = None):
        """Create an unfitted model.

        Args:
            n_edges: Requested quantile edges per feature (``n_edges + 1`` bins).
            n_classes: Number of configs; ``None`` infers ``max(y) + 1`` in ``fit``.
        """
        self.n_edges = n_edges
        self.n_classes = n_classes

    def fit(self, X, y):
        """Compute edges and the per-cell majority table.

        Empty cells take the label of the nearest populated cell (L1 distance in
        bin-index space; first one in flat order on ties).

        Args:
            X (numpy.ndarray): ``(n, n_features)`` raw features.
            y (numpy.ndarray): Integer labels.

        Returns:
            model (BinsModel): ``self``.
        """
        from whdata import quantile_edges
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes = self.n_classes or int(y.max()) + 1
        self.edges = [quantile_edges(X[:, i], self.n_edges) for i in range(X.shape[1])]
        self.shape = tuple(len(e) + 1 for e in self.edges)
        bins = self._bins(X)
        flat = np.ravel_multi_index(tuple(bins.T), self.shape)
        counts = np.zeros((int(np.prod(self.shape)), self.n_classes))
        np.add.at(counts, (flat, y), 1)
        filled = counts.sum(1) > 0
        table = np.where(filled, counts.argmax(1), -1)
        if not filled.all():  # nearest populated cell in bin-index space
            grid = np.stack(np.unravel_index(np.arange(table.size), self.shape), axis=1)
            src = grid[filled]
            for i in np.where(~filled)[0]:
                d = np.abs(src - grid[i]).sum(1)
                table[i] = table[filled][int(np.argmin(d))]
        self.table = table.astype(int)
        return self

    def _bins(self, X):
        """Return the ``(n, n_features)`` bin indices of ``X`` (same rule as ``Lut.bin_index``)."""
        return np.stack([np.searchsorted(e, X[:, i], side="left")
                         for i, e in enumerate(self.edges)], axis=1)

    def predict(self, X):
        """Look up the config of each sample's cell.

        Args:
            X (numpy.ndarray): ``(n, n_features)`` raw features.

        Returns:
            configs (numpy.ndarray): ``(n,)`` integer config indices.
        """
        X = np.asarray(X, dtype=float)
        return self.table[np.ravel_multi_index(tuple(self._bins(X).T), self.shape)]


class MLPModel:
    """Small MLP classifier (PyTorch), deterministic given the seed.

    Attributes:
        name: Model type key (``"mlp"``).
        hidden: Hidden layer width.
        epochs: Training epochs.
        lr: Learning rate.
        weight_decay: Adam weight decay.
        seed: Torch seed.
        n_classes: Number of configs.
        balanced: Whether the loss is class-balanced.
        history: Training loss per epoch.
        scaler: Fitted ``_Scaler``.
        net: Trained ``torch.nn.Module``.
    """
    name = "mlp"

    def __init__(self, hidden: int = 32, epochs: int = 300, lr: float = 1e-2,
                 weight_decay: float = 1e-4, seed: int = 0, n_classes: int | None = None,
                 balanced: bool = True):
        """Create an unfitted model.

        Args:
            hidden: Width of both hidden layers.
            epochs: Full-batch training epochs.
            lr: Adam learning rate.
            weight_decay: Adam weight decay.
            seed: Torch seed.
            n_classes: Number of configs; ``None`` infers ``max(y) + 1`` in ``fit``.
            balanced: Weight the cross-entropy loss by inverse class frequency.
        """
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.weight_decay, self.seed, self.n_classes = weight_decay, seed, n_classes
        self.balanced = balanced
        self.history: list[float] = []

    def _net(self, n_in):
        """Return the ``n_in -> hidden -> hidden -> n_classes`` ReLU network."""
        import torch.nn as nn
        return nn.Sequential(nn.Linear(n_in, self.hidden), nn.ReLU(),
                             nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                             nn.Linear(self.hidden, self.n_classes))

    def fit(self, X, y):
        """Train the network full-batch with Adam on cross-entropy loss.

        Args:
            X (numpy.ndarray): ``(n, n_features)`` raw features.
            y (numpy.ndarray): Integer labels.

        Returns:
            model (MLPModel): ``self``.
        """
        import torch
        torch.set_num_threads(TORCH_THREADS)  # shared 12-thread host (interfaces.md §1)
        y = np.asarray(y, dtype=int)
        self.n_classes = self.n_classes or int(y.max()) + 1
        torch.manual_seed(self.seed)
        self.scaler = _Scaler().fit(X)
        Z = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        T = torch.tensor(y, dtype=torch.long)
        self.net = self._net(Z.shape[1])
        w = None
        if self.balanced:
            c = np.bincount(y, minlength=self.n_classes).astype(float)
            w = torch.tensor(np.where(c > 0, len(y) / (self.n_classes * np.maximum(c, 1)), 0.0),
                             dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.net.train()
        for _ in range(self.epochs):  # full batch: datasets are small
            opt.zero_grad()
            loss = loss_fn(self.net(Z), T)
            loss.backward()
            opt.step()
            self.history.append(float(loss.item()))
        self.net.eval()
        return self

    def predict(self, X):
        """Predict the arg-max config per sample.

        Args:
            X (numpy.ndarray): ``(n, n_features)`` raw features.

        Returns:
            configs (numpy.ndarray): ``(n,)`` integer config indices.
        """
        import torch
        torch.set_num_threads(TORCH_THREADS)
        with torch.no_grad():
            Z = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            return self.net(Z).argmax(1).numpy().astype(int)


class _FeatureTokenTransformer:  # built lazily so importing wh_models needs no torch
    """Lazy builder of the feature-token Transformer network (no torch import at module load)."""
    @staticmethod
    def build(n_in: int, n_classes: int, d_model: int, n_heads: int, n_layers: int):
        """Build the network.

        Each input feature is one token: a linear value embedding of the scalar plus
        a learned per-feature embedding. Tokens pass through ``n_layers`` encoder
        layers (feed-forward width ``2 * d_model``, no dropout) and are mean-pooled
        into a linear head.

        Args:
            n_in: Number of input features (tokens).
            n_classes: Number of output configs.
            d_model: Token embedding width.
            n_heads: Attention heads.
            n_layers: Encoder layers.

        Returns:
            net (torch.nn.Module): Module mapping ``(B, n_in)`` inputs to ``(B, n_classes)``
                logits.
        """
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            """Feature-token Transformer classifier."""
            def __init__(self):
                """Create the embeddings, encoder and head."""
                super().__init__()
                self.value = nn.Linear(1, d_model)
                self.feat = nn.Parameter(torch.randn(n_in, d_model) * 0.02)
                layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                                   dropout=0.0, batch_first=True)
                self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
                self.head = nn.Linear(d_model, n_classes)

            def forward(self, x):                      # x: (B, F)
                """Return ``(B, n_classes)`` logits for ``(B, F)`` standardised features."""
                t = self.value(x.unsqueeze(-1)) + self.feat  # (B, F, d)
                return self.head(self.enc(t).mean(1))

        return Net()


class TransformerModel(MLPModel):
    """Tiny per-window Transformer (features as tokens); mini-batch Adam.

    Pickled without the (locally defined) network class; see ``__getstate__``.

    Attributes:
        name: Model type key (``"transformer"``).
        d_model: Token embedding width.
        n_heads: Attention heads.
        n_layers: Encoder layers.
        batch: Mini-batch size.
    """
    name = "transformer"

    def __init__(self, d_model: int = 16, n_heads: int = 2, n_layers: int = 1,
                 epochs: int = 60, lr: float = 3e-3, batch: int = 1024, seed: int = 0,
                 n_classes: int | None = None, balanced: bool = True):
        """Create an unfitted model.

        Args:
            d_model: Token embedding width.
            n_heads: Attention heads.
            n_layers: Encoder layers.
            epochs: Training epochs.
            lr: Adam learning rate.
            batch: Mini-batch size.
            seed: Torch seed (initialisation and shuffling).
            n_classes: Number of configs; ``None`` infers ``max(y) + 1`` in ``fit``.
            balanced: Weight the cross-entropy loss by inverse class frequency.
        """
        super().__init__(epochs=epochs, lr=lr, seed=seed, n_classes=n_classes, balanced=balanced)
        self.d_model, self.n_heads, self.n_layers, self.batch = d_model, n_heads, n_layers, batch

    def _net(self, n_in):
        """Return the feature-token Transformer for ``n_in`` features."""
        return _FeatureTokenTransformer.build(n_in, self.n_classes, self.d_model, self.n_heads,
                                              self.n_layers)

    def fit(self, X, y):
        """Train with shuffled mini-batch Adam (no weight decay) on cross-entropy loss.

        ``history`` records the sample-weighted mean loss of each epoch.

        Args:
            X (numpy.ndarray): ``(n, n_features)`` raw features.
            y (numpy.ndarray): Integer labels.

        Returns:
            model (TransformerModel): ``self``.
        """
        import torch
        torch.set_num_threads(TORCH_THREADS)
        y = np.asarray(y, dtype=int)
        self.n_classes = self.n_classes or int(y.max()) + 1
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        self.scaler = _Scaler().fit(X)
        Z = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        T = torch.tensor(y, dtype=torch.long)
        self.net = self._net(Z.shape[1])
        w = None
        if self.balanced:
            c = np.bincount(y, minlength=self.n_classes).astype(float)
            w = torch.tensor(np.where(c > 0, len(y) / (self.n_classes * np.maximum(c, 1)), 0.0),
                             dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(T), generator=g)
            tot = 0.0
            for i in range(0, len(T), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(Z[b]), T[b])
                loss.backward()
                opt.step()
                tot += float(loss.item()) * len(b)
            self.history.append(tot / max(len(T), 1))
        self.net.eval()
        return self

    # The network class is local (lazy torch import), so pickle the weights only.
    def __getstate__(self):
        """Return picklable state: the network becomes its CPU ``state_dict`` and input size."""
        st = dict(self.__dict__)
        net = st.pop("net", None)
        if net is not None:
            st["_net_state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            st["_n_in"] = int(net.feat.shape[0])
        return st

    def __setstate__(self, st):
        """Restore state, rebuilding the network from the pickled weights if present."""
        sd, n_in = st.pop("_net_state", None), st.pop("_n_in", None)
        self.__dict__.update(st)
        if sd is not None:
            self.net = self._net(n_in)
            self.net.load_state_dict(sd)
            self.net.eval()

    def predict(self, X, batch: int = 65536):
        """Predict the arg-max config per sample, in batches.

        Args:
            X (numpy.ndarray): ``(n, n_features)`` raw features.
            batch: Rows per forward pass.

        Returns:
            configs (numpy.ndarray): ``(n,)`` integer config indices.
        """
        import torch
        torch.set_num_threads(TORCH_THREADS)
        out = []
        with torch.no_grad():
            Z = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            for i in range(0, len(Z), batch):
                out.append(self.net(Z[i:i + batch]).argmax(1).numpy())
        return np.concatenate(out).astype(int) if out else np.zeros(0, dtype=int)


def make_model(kind: str, n_classes: int, **kw):
    """Create an unfitted model of the given type.

    Only the keyword arguments relevant to that type are passed on: ``seed`` and
    ``t_epochs`` (as ``epochs``, when truthy) for ``transformer``; ``hidden``,
    ``epochs``, ``lr``, ``seed`` for ``mlp``; ``max_depth``, ``min_leaf`` for
    ``tree``; ``k`` for ``knn``; ``n_edges`` for ``bins``. Others are ignored.

    Args:
        kind: One of ``MODEL_TYPES``.
        n_classes: Number of configs.
        **kw (Any): Hyper-parameters (see above).

    Returns:
        model (object): The model instance.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind == "transformer":
        return TransformerModel(n_classes=n_classes, **{k: v for k, v in kw.items()
                                                        if k == "seed"},
                                **({"epochs": kw["t_epochs"]} if kw.get("t_epochs") else {}))
    if kind == "mlp":
        return MLPModel(n_classes=n_classes, **{k: v for k, v in kw.items()
                                                 if k in ("hidden", "epochs", "lr", "seed")})
    if kind == "tree":
        return TreeModel(n_classes=n_classes, **{k: v for k, v in kw.items()
                                                  if k in ("max_depth", "min_leaf")})
    if kind == "knn":
        return KNNModel(n_classes=n_classes, **{k: v for k, v in kw.items() if k == "k"})
    if kind == "bins":
        return BinsModel(n_classes=n_classes, **{k: v for k, v in kw.items() if k == "n_edges"})
    raise ValueError(f"unknown model type {kind!r}; choose from {MODEL_TYPES}")
