# Market Microstructure Simulator

An average simulator.

## Code Layout

```text
src/
├── engine/
│   ├── exchange/
│   ├── matching/
│   ├── orderbook/
│   └── orders/
├── simulation/
│   ├── agents/
│   ├── events/
│   ├── portfolio/
│   └── simulator/
├── analytics/
├── ui/
└── tests/
```

## Development

```bash
npm install
npm run test
npm run build
```

## Dataset export

```bash
npm run dataset -- mlp
npm run dataset -- cnn
```

Phase 1 writes raw public features to versioned `dataset.jsonl` and `metadata.json`
under `datasets/training/<layout>/`. Each row includes trajectory seed, observation
step and a direction class. Use `--output` with a new directory for repeat runs;
existing directories are protected from overwriting. See the
[dataset contract and CLI options](docs/dataset-contract-v1.md).

The obsolete normalized datasets and TypeScript training command have been removed;
training now uses the Python stack below.

## Python training (Phase 2)

```bash
uv sync --locked
uv run python -m ml.training.train --dataset datasets/training/mlp --output models/checkpoints/baseline.pt
uv run pytest -q
```

The Python package in `ml/` streams the raw exports, trains a small registered
PyTorch MLP, and saves a versioned checkpoint with train-only preprocessing.
Validation uses independent trajectories; early stopping restores the best epoch.
Use a new checkpoint path for each run and `--help` for configuration options.
Python CNN training (Phase 3) is deferred. Local details are in `docs/python-training-phase-2.md`;
`docs/` and generated Python checkpoints stay outside version control.

## Python inference (Phase 4)

```bash
uv run python -m ml.inference.server --model market-side-v1=models/checkpoints/baseline.pt
```

The process loads each checkpoint once and reads one JSON request per stdin line:

```json
{"id":1,"type":"predict","model":"market-side-v1","input":[0.1,0.2]}
```

Replace the example input with the full raw feature vector described by the checkpoint's
metadata. The server applies saved preprocessing, then returns
`{"id":1,"type":"prediction","prediction":[...]}` containing **logits** in the saved
class order (`BUY`, `SELL`, `HOLD` for dataset contract v1). Errors return
`{"id":1,"type":"error","error":"..."}`. IDs are non-negative JavaScript-safe integers;
unreadable or invalid IDs return `null`. Requests allow exactly `id`, `type`, `model`
and `input`, with finite float32-compatible numbers and the exact feature count.

Repeat `--model ALIAS=PATH` for multiple deployments, or pass `--registry registry.json`
with an object mapping aliases to checkpoint paths. Paths in a registry file are relative
to that file; CLI paths are relative to the working directory. Aliases are independent
of architecture and feature-layout IDs. Architecture construction uses the existing
Python model registry and saved configuration; adding a model does not change this server.
Custom entry points can register an architecture before calling `ml.inference.server.main()`.

Responses flush immediately; stdout is JSONL only and diagnostics go to stderr.
Startup fails with a nonzero exit status if any checkpoint cannot load. Invalid requests
do not terminate the process. Each UTF-8 line has a 1 MiB limit including its newline;
oversized lines are discarded as a whole before the next request. EOF exits cleanly.
The default is CPU inference with one thread (`--num-threads` is configurable).

The browser still uses the native inference path until the TypeScript subprocess client
and simulator integration in Phases 5–6. Its live `poc` runtime and MLP checkpoint remain;
unused training code, schedules, legacy datasets and CNN checkpoint artifacts are removed.
Local implementation notes are in `docs/python-inference-phase-4.md`.

## Model extensibility

Models, feature layouts and normalizers have separate registration
APIs. Shared pipelines discover available registrations; runtime defaults live in
`src/simulation/poc/runtime-presets.ts`. See [adding models and feature layouts](docs/model-extensibility.md)
for the extension points, custom entry points and agent configuration.

## Features

A number of typical market participant behaviors are represented in this project. This includes market making, retail trading, momentum trading, and imbalance trading. Each agent operates on the same simulated limit order book while using different strategies and observable market information to make trading decisions.

Agents interact with the market through a shared observable market context. This context is intentionally limited to information that would be reasonably available to a participant trading on the simulated exchange (current and previous midPrices, currently resting orders, recent trades, etc.), preventing agents from accessing privileged information about other participants. 

## Goals

The long-term goal is to use the simulator as an testing environment for increasingly sophisticated trading agents. In particular, the project will eventually incorporate deep learning-based agents, with performance evaluated against the other simulated participants.

The [Phase 0 ML migration audit](docs/ml-migration-phase-0.md) documents the current
data, training and inference flows, the proposed TypeScript/Python boundary, and
decisions required before implementation of later phases.

## Deployment

A production deployment can be accessed [here](https://market-gpsakura.vercel.app)
