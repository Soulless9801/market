import { describe, expect, it } from "vitest";

import { Exchange } from "../engine";
import { MarketMakerAgent, RetailTraderAgent, Simulator } from "../simulation";

describe("Phase 2 simulation layer", () => {
	it("market maker agent emits both bid and ask limit orders around the midprice", () => {
		const exchange = new Exchange();
		const agent = new MarketMakerAgent("mm-1", {
			referencePrice: 100,
			spread: 2,
			quantity: 10,
		});

		const orders = agent.step({
			clock: 1,
			snapshot: exchange.getOrderBookSnapshot(),
			midPrice: 100,
			tradeCount: 0,
		});

		expect(orders).toHaveLength(2);
		expect(orders[0]).toMatchObject({
			participantId: "mm-1",
			side: "BUY",
			type: "LIMIT",
			quantity: 10,
			price: 99,
		});
		expect(orders[1]).toMatchObject({
			participantId: "mm-1",
			side: "SELL",
			type: "LIMIT",
			quantity: 10,
			price: 101,
		});

		const reports = orders.map((order) => exchange.submitOrder(order));
		expect(reports.every((report) => report.status === "RESTING")).toBe(true);
	});

	it("simulation loop processes deterministic steps and records trade events", () => {
		const marketMaker = new MarketMakerAgent("mm-1", {
			referencePrice: 100,
			spread: 2,
			quantity: 10,
		});
		const retail = new RetailTraderAgent("retail-1", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 7,
			maxPriceOffset: 1,
			bias: "BUY",
		});

		const simulator = new Simulator({
			agents: [marketMaker, retail],
			referencePrice: 100,
		});

		const firstStep = simulator.runStep();
		const secondStep = simulator.runStep();

		expect(firstStep.events.some((event) => event.type === "agent-step")).toBe(true);
		expect(secondStep.events.some((event) => event.type === "trade")).toBe(true);
		expect(simulator.getTradeHistory()).toHaveLength(1);
		expect(simulator.getEvents()).toEqual(
			expect.arrayContaining([expect.objectContaining({ type: "trade" })]),
		);
	});
});
