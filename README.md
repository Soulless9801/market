# Market Microstructure Simulator

An exchange simulation project focused on order-driven market mechanics, not price prediction.

## Current scope (Phase 1)

Implemented:

- strongly-typed order and trade models
- deterministic central limit order book
- ordered-statistic-tree price level index (logarithmic best-price insert/remove/select)
- price-time priority matching
- limit and market order handling
- order cancellation
- unit tests for core matching behavior

## Architecture (engine-first)

The simulation engine is separated from React UI:

```text
UI -> Simulation Engine -> Exchange -> Matching Engine -> Order Book
```

Code layout:

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
