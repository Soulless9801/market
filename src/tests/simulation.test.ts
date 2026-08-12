import { describe, expect, it } from "vitest";

import { Exchange } from "../engine";
import { buildDefaultAgents, MarketMakerAgent, RetailTraderAgent, Simulator } from "../simulation";

describe("Phase 2 simulation layer", () => {
	
	it("builds the default market composition with one market maker and two retail traders", () => {
		const agents = buildDefaultAgents(7, 100);

		expect(agents).toHaveLength(4);
		expect(agents[0]).toBeInstanceOf(MarketMakerAgent);
		expect(agents[1]).toBeInstanceOf(RetailTraderAgent);
		expect(agents[2]).toBeInstanceOf(RetailTraderAgent);
		expect(agents[3]).toBeInstanceOf(RetailTraderAgent);
		expect(agents.map((agent) => agent.id)).toEqual(["mm-1", "retail-1", "retail-2", "retail-3"]);
	});

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
		expect(simulator.getTradeHistory().length).toBeGreaterThanOrEqual(1);
		expect(simulator.getEvents()).toEqual(
			expect.arrayContaining([expect.objectContaining({ type: "trade" })]),
		);
	});

	it("passive buy orders do not cross the best ask", () => {
		const exchange = new Exchange();
		exchange.submitOrder({
			participantId: "mm-1",
			side: "BUY",
			type: "LIMIT",
			price: 99.5,
			quantity: 8,
		});
		exchange.submitOrder({
			participantId: "mm-1",
			side: "SELL",
			type: "LIMIT",
			price: 100.5,
			quantity: 8,
		});

		const agent = new RetailTraderAgent("retail-1", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 7,
			bias: "BUY",
			executionStyle: "PASSIVE",
		});

		const [order] = agent.step({
			clock: 1,
			snapshot: exchange.getOrderBookSnapshot(),
			midPrice: 100,
			tradeCount: 0,
		});
		const report = exchange.submitOrder(order);

		expect(order.type).toBe("LIMIT");
		if (order.type !== "LIMIT") {
			throw new Error("Expected a limit order");
		}
		expect(order.side).toBe("BUY");
		expect(order.price).toBeLessThanOrEqual(99.5);
		expect(report.status).toBe("RESTING");
		expect(report.trades).toHaveLength(0);
	});

	it("passive sell orders do not cross the best bid", () => {
		const exchange = new Exchange();
		exchange.submitOrder({
			participantId: "mm-1",
			side: "BUY",
			type: "LIMIT",
			price: 99.5,
			quantity: 8,
		});
		exchange.submitOrder({
			participantId: "mm-1",
			side: "SELL",
			type: "LIMIT",
			price: 100.5,
			quantity: 8,
		});

		const agent = new RetailTraderAgent("retail-2", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 11,
			bias: "SELL",
			executionStyle: "PASSIVE",
		});

		const [order] = agent.step({
			clock: 1,
			snapshot: exchange.getOrderBookSnapshot(),
			midPrice: 100,
			tradeCount: 0,
		});
		const report = exchange.submitOrder(order);

		expect(order.type).toBe("LIMIT");
		if (order.type !== "LIMIT") {
			throw new Error("Expected a limit order");
		}
		expect(order.side).toBe("SELL");
		expect(order.price).toBeGreaterThanOrEqual(100.5);
		expect(report.status).toBe("RESTING");
		expect(report.trades).toHaveLength(0);
	});

	it("aggressive buy orders can consume ask liquidity", () => {
		const exchange = new Exchange();
		const restingAsk = exchange.submitOrder({
			participantId: "mm-1",
			side: "SELL",
			type: "LIMIT",
			price: 100.5,
			quantity: 8,
		});
		expect(restingAsk.status).toBe("RESTING");

		const agent = new RetailTraderAgent("retail-3", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 3,
			bias: "BUY",
			executionStyle: "AGGRESSIVE",
		});

		const [order] = agent.step({
			clock: 1,
			snapshot: exchange.getOrderBookSnapshot(),
			midPrice: 100,
			tradeCount: 0,
		});
		const report = exchange.submitOrder(order);

		expect(report.status).toBe("RESTING");
		expect(report.filledQuantity).toBeGreaterThan(0);
		expect(report.trades).toHaveLength(1);
	});

	it("aggressive sell orders can consume bid liquidity", () => {
		const exchange = new Exchange();
		const restingBid = exchange.submitOrder({
			participantId: "mm-1",
			side: "BUY",
			type: "LIMIT",
			price: 99.5,
			quantity: 8,
		});
		expect(restingBid.status).toBe("RESTING");

		const agent = new RetailTraderAgent("retail-4", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 5,
			bias: "SELL",
			executionStyle: "AGGRESSIVE",
		});

		const [order] = agent.step({
			clock: 1,
			snapshot: exchange.getOrderBookSnapshot(),
			midPrice: 100,
			tradeCount: 0,
		});
		const report = exchange.submitOrder(order);

		expect(report.status).toBe("RESTING");
		expect(report.filledQuantity).toBeGreaterThan(0);
		expect(report.trades).toHaveLength(1);
	});

	it("random execution style stays deterministic for a fixed seed", () => {
		const context = {
			clock: 1,
			snapshot: { bids: [], asks: [] },
			midPrice: 100,
			tradeCount: 0,
		};
		const first = new RetailTraderAgent("retail-5", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 19,
			bias: "RANDOM",
			executionStyle: "RANDOM",
		}).step(context);
		const second = new RetailTraderAgent("retail-6", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 19,
			bias: "RANDOM",
			executionStyle: "RANDOM",
		}).step(context);

		expect(first[0].type).toBe("LIMIT");
		expect(second[0].type).toBe("LIMIT");
		if (first[0].type !== "LIMIT" || second[0].type !== "LIMIT") {
			throw new Error("Expected limit orders");
		}
		expect(first[0]).toMatchObject({
			quantity: second[0].quantity,
			side: second[0].side,
			price: second[0].price,
		});
	});

	it("random side bias can produce both buy and sell orders", () => {
		const context = {
			clock: 1,
			snapshot: { bids: [], asks: [] },
			midPrice: 100,
			tradeCount: 0,
		};
		const sides = [1, 682].map((seed) => new RetailTraderAgent(`retail-${seed}`, {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed,
			bias: "RANDOM",
			executionStyle: "RANDOM",
		}).step(context)[0].side);

		expect(sides).toContain("BUY");
		expect(sides).toContain("SELL");
	});

	it("reset restores the simulator to its initial state", () => {
		const simulator = new Simulator({
			agents: [
				new MarketMakerAgent("mm-1", {
					referencePrice: 100,
					spread: 2,
					quantity: 10,
				}),
			],
			referencePrice: 100,
		});

		simulator.runStep();
		simulator.reset();

		expect(simulator.getClock()).toBe(0);
		expect(simulator.getTradeHistory()).toHaveLength(0);
		expect(simulator.getEvents()).toHaveLength(0);
		expect(simulator.getParticpantPortfolios()).toEqual([
			expect.objectContaining({ 
				participantId: "mm-1", 
				cash: 100000,
				equity: 100000,
				inventory: 0,
				marketValue: 0,
				midPrice: 100,
				pnl: 0,
			}),
		]);
	});
});
