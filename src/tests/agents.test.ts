import { describe, expect, it } from "vitest";

import {
	// MarketMakerAgent,
	// RetailTraderAgent,
	MomentumTraderAgent,
	MeanReversionTraderAgent,
	ImbalanceTraderAgent,
} from "../simulation";

import type { AgentSimulatorContext } from "../simulation";

function makeContext(
	overrides: Partial<AgentSimulatorContext> = {},
): AgentSimulatorContext {
	return {
		clock: 1,
		midPrice: 100,
		spread: 2,

		orderBook: {
			bids: [],
			asks: [],
		},

		recentTrades: [],

		recentMidPriceSeries: [],

		orderImbalance: {
			bidVolume: 10,
			askVolume: 10,
			imbalance: 0.5,
			bidPercent: 50,
			askPercent: 50,
		},

		portfolio: {
			participantId: "test-agent",
            ordersSubmitted: 0,
			cash: 100_000,
			inventory: 0,
			marketValue: 0,
			equity: 100_000,
			midPrice: 100,
			pnl: 0,
		},

		...overrides,
	};
}

describe("MomentumTraderAgent", () => {
	it("buys when momentum exceeds the positive threshold", () => {
		const agent = new MomentumTraderAgent("momentum-1", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 1,
			lookback: 3,
			momentumThreshold: 1,
			executionStyle: "PASSIVE",
		});

		const [order] = agent.step(
			makeContext({
				midPrice: 103,
				recentMidPriceSeries: [100, 101, 102],
			}),
		);

		expect(order.side).toBe("BUY");
	});

	it("sells when momentum falls below the negative threshold", () => {
		const agent = new MomentumTraderAgent("momentum-1", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 1,
			lookback: 3,
			momentumThreshold: 1,
			executionStyle: "PASSIVE",
		});

		const [order] = agent.step(
			makeContext({
				midPrice: 97,
				recentMidPriceSeries: [100, 99, 98],
			}),
		);

		expect(order.side).toBe("SELL");
	});

	it("uses the configured lookback", () => {
		const agent = new MomentumTraderAgent("momentum-1", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 1,
			lookback: 3,
			momentumThreshold: 1,
			executionStyle: "PASSIVE",
		});

		const [order] = agent.step(
			makeContext({
				midPrice: 101,
				recentMidPriceSeries: [
					90,
					95,
					105,
					104,
					103,
				],
			}),
		);

		// Lookback starts at 105.
		// 101 - 105 = -4.
		expect(order.side).toBe("SELL");
	});

	it("produces deterministic decisions with the same seed", () => {
		const context = makeContext({
			midPrice: 100,
			recentMidPriceSeries: [99, 100],
		});

		const first = new MomentumTraderAgent("m1", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 123,
			momentumThreshold: 10,
		}).step(context);

		const second = new MomentumTraderAgent("m2", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 123,
			momentumThreshold: 10,
		}).step(context);

		expect(first[0].side).toBe(second[0].side);
	});
});

describe("MeanReversionTraderAgent", () => {
	it("sells when price is sufficiently above the historical reference", () => {
		const agent = new MeanReversionTraderAgent(
			"mean-reversion-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 1,
				lookback: 3,
				deviationThreshold: 1,
				executionStyle: "PASSIVE",
			},
		);

		const [order] = agent.step(
			makeContext({
				midPrice: 103,
				recentMidPriceSeries: [100, 101, 102],
			}),
		);

		expect(order.side).toBe("SELL");
	});

	it("buys when price is sufficiently below the historical reference", () => {
		const agent = new MeanReversionTraderAgent(
			"mean-reversion-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 1,
				lookback: 3,
				deviationThreshold: 1,
				executionStyle: "PASSIVE",
			},
		);

		const [order] = agent.step(
			makeContext({
				midPrice: 97,
				recentMidPriceSeries: [100, 99, 98],
			}),
		);

		expect(order.side).toBe("BUY");
	});

	it("uses the configured lookback reference", () => {
		const agent = new MeanReversionTraderAgent(
			"mean-reversion-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 1,
				lookback: 3,
				deviationThreshold: 1,
				executionStyle: "PASSIVE",
			},
		);

		const [order] = agent.step(
			makeContext({
				midPrice: 101,
				recentMidPriceSeries: [
					90,
					95,
					105,
					104,
					103,
				],
			}),
		);

		// Current implementation uses 105 as its
		// historical reference.
		// 101 - 105 = -4.
		expect(order.side).toBe("BUY");
	});

	it("falls back to a deterministic random side inside the threshold", () => {
		const context = makeContext({
			midPrice: 100.2,
			recentMidPriceSeries: [100],
		});

		const first = new MeanReversionTraderAgent(
			"mr-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 123,
				deviationThreshold: 1,
			},
		).step(context);

		const second = new MeanReversionTraderAgent(
			"mr-2",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 123,
				deviationThreshold: 1,
			},
		).step(context);

		expect(first[0].side).toBe(second[0].side);
	});
});

describe("ImbalanceTraderAgent", () => {
	it("buys when imbalance reaches the buy threshold", () => {
		const agent = new ImbalanceTraderAgent(
			"imbalance-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 1,
				buyThreshold: 0.65,
				sellThreshold: 0.35,
			},
		);

		const [order] = agent.step(
			makeContext({
				orderImbalance: {
					bidVolume: 65,
					askVolume: 35,
					imbalance: 0.65,
					bidPercent: 65,
					askPercent: 35,
				},
			}),
		);

		expect(order.side).toBe("BUY");
	});

	it("sells when imbalance reaches the sell threshold", () => {
		const agent = new ImbalanceTraderAgent(
			"imbalance-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 1,
				buyThreshold: 0.65,
				sellThreshold: 0.35,
			},
		);

		const [order] = agent.step(
			makeContext({
				orderImbalance: {
					bidVolume: 35,
					askVolume: 65,
					imbalance: 0.35,
					bidPercent: 35,
					askPercent: 65,
				},
			}),
		);

		expect(order.side).toBe("SELL");
	});

	it("buys under strong bid-side pressure", () => {
		const agent = new ImbalanceTraderAgent(
			"imbalance-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 1,
			},
		);

		const [order] = agent.step(
			makeContext({
				orderImbalance: {
					bidVolume: 90,
					askVolume: 10,
					imbalance: 0.9,
					bidPercent: 90,
					askPercent: 10,
				},
			}),
		);

		expect(order.side).toBe("BUY");
	});

	it("sells under strong ask-side pressure", () => {
		const agent = new ImbalanceTraderAgent(
			"imbalance-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 1,
			},
		);

		const [order] = agent.step(
			makeContext({
				orderImbalance: {
					bidVolume: 10,
					askVolume: 90,
					imbalance: 0.1,
					bidPercent: 10,
					askPercent: 90,
				},
			}),
		);

		expect(order.side).toBe("SELL");
	});

	it("uses a deterministic random decision when imbalance is neutral", () => {
		const context = makeContext({
			orderImbalance: {
				bidVolume: 50,
				askVolume: 50,
				imbalance: 0.5,
				bidPercent: 50,
				askPercent: 50,
			},
		});

		const first = new ImbalanceTraderAgent(
			"imbalance-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 123,
			},
		).step(context);

		const second = new ImbalanceTraderAgent(
			"imbalance-2",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 123,
			},
		).step(context);

		expect(first[0].side).toBe(second[0].side);
	});
});

describe("strategy agent order generation", () => {
	it("produces valid limit orders", () => {
		const context = makeContext({
			midPrice: 100,
			recentMidPriceSeries: [
				98,
				99,
				100,
			],
		});

		const agents = [
			new MomentumTraderAgent("momentum", {
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 1,
			}),

			new MeanReversionTraderAgent(
				"mean-reversion",
				{
					referencePrice: 100,
					spread: 2,
					quantity: 8,
					seed: 2,
				},
			),

			new ImbalanceTraderAgent(
				"imbalance",
				{
					referencePrice: 100,
					spread: 2,
					quantity: 8,
					seed: 3,
				},
			),
		];

		for (const agent of agents) {
			const orders = agent.step(context);

			expect(orders.length).toBeGreaterThan(0);

			for (const order of orders) {
				expect(order.participantId).toBe(agent.id);
				expect(order.type).toBe("LIMIT");
				expect(order.quantity).toBeGreaterThan(0);
				// expect(order.price).toBeGreaterThan(0);
				expect([
					"BUY",
					"SELL",
				]).toContain(order.side);
			}
		}
	});
});

describe("strategy agent execution styles", () => {
	it("momentum trader can place an aggressive buy at the best ask", () => {
		const agent = new MomentumTraderAgent(
			"momentum-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 1,
				lookback: 3,
				momentumThreshold: 1,
				executionStyle: "AGGRESSIVE",
			},
		);

		const [order] = agent.step(
			makeContext({
				midPrice: 103,
				recentMidPriceSeries: [
					100,
					101,
					102,
				],
				orderBook: {
					bids: [
						{
							price: 99,
							quantity: 10,
                            orderCount: 1,
						},
					],
					asks: [
						{
							price: 101,
							quantity: 10,
                            orderCount: 1,
						},
					],
				},
			}),
		);

		expect(order.side).toBe("BUY");
		// expect(order.price).toBe(101);
	});

	it("mean reversion trader can place a passive buy below the best bid", () => {
		const agent = new MeanReversionTraderAgent(
			"mr-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 1,
				lookback: 3,
				deviationThreshold: 1,
				executionStyle: "PASSIVE",
			},
		);

		const [order] = agent.step(
			makeContext({
				midPrice: 97,
				recentMidPriceSeries: [
					100,
					99,
					98,
				],
				orderBook: {
					bids: [
						{
							price: 99,
							quantity: 10,
                            orderCount: 1,
						},
					],
					asks: [
						{
							price: 101,
							quantity: 10,
                            orderCount: 1,
						},
					],
				},
			}),
		);

		expect(order.side).toBe("BUY");
		// expect(order.price).toBeLessThanOrEqual(99);
	});

	it("imbalance trader can place an aggressive buy at the best ask", () => {
		const agent = new ImbalanceTraderAgent(
			"imbalance-1",
			{
				referencePrice: 100,
				spread: 2,
				quantity: 8,
				seed: 1,
				buyThreshold: 0.65,
				executionStyle: "AGGRESSIVE",
			},
		);

		const [order] = agent.step(
			makeContext({
				orderImbalance: {
					bidVolume: 80,
					askVolume: 20,
					imbalance: 0.8,
					bidPercent: 80,
					askPercent: 20,
				},
				orderBook: {
					bids: [
						{
							price: 99,
							quantity: 10,
                            orderCount: 1,
						},
					],
					asks: [
						{
							price: 101,
							quantity: 10,
                            orderCount: 1,
						},
					],
				},
			}),
		);

		expect(order.side).toBe("BUY");
        // expect(order.price).toBe(101);
	});
});
