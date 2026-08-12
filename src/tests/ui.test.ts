import { describe, expect, it } from "vitest";

import type { OrderBookSnapshot, TradeEvent } from "../engine";
import {
	buildBookRows,
	buildMarketViewModel,
	buildTradeTape,
} from "../ui";
import { calculateOrderImbalance, calculateSpread } from "../engine";

describe("Phase 3 view-model helpers", () => {
	it("calculates spread from best bid and best ask", () => {
		const snapshot: OrderBookSnapshot = {
			bids: [{ price: 100.0, quantity: 20, orderCount: 1 }],
			asks: [{ price: 100.5, quantity: 15, orderCount: 1 }],
		};

		expect(calculateSpread(snapshot)).toBe(0.5);
	});

	it("computes order-book imbalance from quantity across sides", () => {
		const snapshot: OrderBookSnapshot = {
			bids: [
				{ price: 100.0, quantity: 60, orderCount: 1 },
				{ price: 99.9, quantity: 20, orderCount: 1 },
			],
			asks: [{ price: 100.1, quantity: 40, orderCount: 1 }],
		};

		const imbalance = calculateOrderImbalance(snapshot);
		expect(imbalance.bidVolume).toBe(80);
		expect(imbalance.askVolume).toBe(40);
		expect(imbalance.imbalance).toBe(0.6666666666666666);
		expect(imbalance.bidPercent).toBe(66.7);
	});

	it("builds trade tape entries and market view model metadata", () => {
		const snapshot: OrderBookSnapshot = {
			bids: [{ price: 100.0, quantity: 20, orderCount: 1 }],
			asks: [{ price: 100.1, quantity: 15, orderCount: 1 }],
		};
		const trades: TradeEvent[] = [
			{
				tradeId: "t-1",
				price: 100.0,
				quantity: 10,
				buyOrderId: "buy-1",
				buyerParticipantId: "retail-1",
				sellOrderId: "sell-1",
				sellerParticipantId: "mm-1",
				makerOrderId: "sell-1",
				makerParticipantId: "mm-1",
				takerOrderId: "buy-1",
				takerParticipantId: "retail-1",
				aggressorSide: "BUY",
				timestamp: 3,
			},
			{
				tradeId: "t-2",
				price: 100.1,
				quantity: 5,
				buyOrderId: "buy-2",
				buyerParticipantId: "retail-2",
				sellOrderId: "sell-2",
				sellerParticipantId: "mm-2",
				makerOrderId: "sell-2",
				makerParticipantId: "mm-2",
				takerOrderId: "buy-2",
				takerParticipantId: "retail-2",
				aggressorSide: "SELL",
				timestamp: 4,
			},
		];

		const viewModel = buildMarketViewModel(snapshot, trades, [], 5, [100.05]);
		expect(viewModel.tradeCount).toBe(2);
		expect(viewModel.totalVolume).toBe(15);
		expect(viewModel.spread).toBeCloseTo(0.1);
		expect(buildTradeTape(trades)).toHaveLength(2);
		expect(buildBookRows(snapshot).asks[0]?.price).toBe(100.1);
	});
});
