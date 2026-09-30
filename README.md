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

The existing normalized datasets and TypeScript training command remain available
as the legacy baseline. Python training is a later phase.

## Model extensibility

Models, feature layouts, normalizers and legacy datasets have separate registration
APIs. Shared pipelines discover available registrations; runtime defaults live in
`src/simulation/ml/runtime-presets.ts`. See [adding models and feature layouts](docs/model-extensibility.md)
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
