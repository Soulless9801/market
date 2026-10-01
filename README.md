# Market Microstructure Simulator

An average simulator.

## Development

```bash
npm install
npm run test
npm run build
```

## Dataset

```bash
npm run dataset -- mlp
npm run dataset -- cnn
npm run dataset -- <model>
```

## Python

Setup
```bash
uv sync --locked
uv run pytest -q
```

Training
```bash
uv run python -m ml.training.train --dataset datasets/training/mlp --output models/checkpoints/baseline.pt
```

## Features

A number of typical market participant behaviors are represented in this project. This includes market making, retail trading, momentum trading, and imbalance trading. Each agent operates on the same simulated limit order book while using different strategies and observable market information to make trading decisions.

Agents interact with the market through a shared observable market context. This context is intentionally limited to information that would be reasonably available to a participant trading on the simulated exchange (current and previous midPrices, currently resting orders, recent trades, etc.), preventing agents from accessing privileged information about other participants. 

## Goals

The long-term goal is to use the simulator as an testing environment for increasingly sophisticated trading agents. In particular, the project will eventually incorporate deep learning-based agents, with performance evaluated against the other simulated participants.

## Deployment

A production deployment can be accessed [here](https://market-gpsakura.vercel.app)
