import { describe, expect, it } from "vitest";

import { Exchange } from "../engine";
import {
	buildDefaultAgents,
	MarketMakerAgent,
	RetailTraderAgent,
	Simulator,
	PortfolioManager,
	MomentumTraderAgent,
	ImbalanceTraderAgent,
	MeanReversionTraderAgent,
} from "../simulation";
import type { ObservableSimulatorContext, AgentSimulatorContext } from "../simulation";

function createObservableContext(
	exchange: Exchange,
	overrides: Partial<ObservableSimulatorContext> = {},
): ObservableSimulatorContext {
	const orderBook = exchange.getOrderBookSnapshot();

	const bestBid = orderBook.bids[0]?.price;
	const bestAsk = orderBook.asks[0]?.price;

	const midPrice =
		bestBid !== undefined && bestAsk !== undefined
			? (bestBid + bestAsk) / 2
			: 100;

	const spread =
		bestBid !== undefined && bestAsk !== undefined
			? bestAsk - bestBid
			: 0;

	const bidVolume = orderBook.bids.reduce(
		(sum, level) => sum + level.quantity,
		0,
	);

	const askVolume = orderBook.asks.reduce(
		(sum, level) => sum + level.quantity,
		0,
	);

	const totalVolume = bidVolume + askVolume;

	const imbalance =
		totalVolume === 0
			? 0
			: bidVolume / totalVolume;

	return {
		clock: 1,
		midPrice,
		spread,
		orderBook,
		recentTrades: [],
		recentMidPriceSeries: [],
		orderImbalance: {
			bidVolume,
			askVolume,
			imbalance,
			bidPercent: imbalance * 100,
			askPercent: (1 - imbalance) * 100,
		},
		...overrides,
	};
}

function createAgentContext(
	agentId: string,
	observableContext: ObservableSimulatorContext,
	portfolioManager: PortfolioManager,
	overrides: Partial<AgentSimulatorContext> = {},
): AgentSimulatorContext {
	const portfolioSnapshot = portfolioManager.getPortfolioSnapshot(agentId, observableContext.orderBook);
	if (!portfolioSnapshot) {
		throw new Error(`Portfolio snapshot for agent ${agentId} not found`);
	}
	return {
		portfolio: portfolioSnapshot,
		...observableContext,
		...overrides,
	}
}


describe("simulation", () => {
	it("builds the default market composition with one market maker and three retail traders", () => {
		const agents = buildDefaultAgents(7, 100);

		expect(agents).toHaveLength(7);
		expect(agents[0]).toBeInstanceOf(MarketMakerAgent);
		expect(agents[1]).toBeInstanceOf(RetailTraderAgent);
		expect(agents[2]).toBeInstanceOf(RetailTraderAgent);
		expect(agents[3]).toBeInstanceOf(RetailTraderAgent);
		expect(agents[4]).toBeInstanceOf(MomentumTraderAgent);
		expect(agents[5]).toBeInstanceOf(MeanReversionTraderAgent);
		expect(agents[6]).toBeInstanceOf(ImbalanceTraderAgent);

		expect(agents.map((agent) => agent.id)).toEqual([
			"mm-1",
			"retail-1",
			"retail-2",
			"retail-3",
			"momentum-1",
			"mean-reversion-1",
			"imbalance-1"
		]);
	});

	it("market maker agent emits both bid and ask limit orders around the midprice", () => {
		const exchange = new Exchange();

		const agent = new MarketMakerAgent("mm-1", {
			referencePrice: 100,
			spread: 2,
			quantity: 10,
		});

		const portfolioManager = new PortfolioManager();
		portfolioManager.createPortfolio("mm-1", 100000);

		const context = createObservableContext(exchange, {
			midPrice: 100,
			spread: 2,
		});

		const agentContext = createAgentContext("mm-1", context, portfolioManager);

		const orders = agent.step(agentContext);

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

		const reports = orders.map((order) =>
			exchange.submitOrder(order),
		);

		expect(
			reports.every(
				(report) => report.status === "RESTING",
			),
		).toBe(true);
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
			executionStyle: "AGGRESSIVE",
		});

		const simulator = new Simulator({
			agents: [marketMaker, retail],
			referencePrice: 100,
		});

		const firstStep = simulator.runStep();
		const secondStep = simulator.runStep();

		expect(
			firstStep.events.some(
				(event) => event.type === "agent-step",
			),
		).toBe(true);

		expect(
			secondStep.events.some(
				(event) => event.type === "trade",
			),
		).toBe(true);

		expect(
			simulator.getLimitedTradeHistory().length,
		).toBeGreaterThanOrEqual(1);

		expect(simulator.getEvents()).toEqual(
			expect.arrayContaining([
				expect.objectContaining({
					type: "trade",
				}),
			]),
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

		const portfolioManager = new PortfolioManager();
		portfolioManager.createPortfolio("retail-1", 100000);

		const context = createObservableContext(exchange, {
			midPrice: 100,
			spread: 1,
		});

		const agentContext = createAgentContext("retail-1", context, portfolioManager);

		const [order] = agent.step(agentContext);
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

		const context = createObservableContext(exchange, {
			midPrice: 100,
			spread: 1,
		});

		const portfolioManager = new PortfolioManager();
		portfolioManager.createPortfolio("retail-2", 100000);

		const agentContext = createAgentContext("retail-2", context, portfolioManager);

		const [order] = agent.step(agentContext);
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

		const context = createObservableContext(exchange, {
			midPrice: 100,
			spread: 0.5,
		});

		const portfolioManager = new PortfolioManager();
		portfolioManager.createPortfolio("retail-3", 100000);

		const agentContext = createAgentContext("retail-3", context, portfolioManager);

		const [order] = agent.step(agentContext);
		const report = exchange.submitOrder(order);

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

		const context = createObservableContext(exchange, {
			midPrice: 100,
			spread: 0.5,
		});

		const portfolioManager = new PortfolioManager();
		portfolioManager.createPortfolio("retail-4", 100000);

		const agentContext = createAgentContext("retail-4", context, portfolioManager);

		const [order] = agent.step(agentContext);
		const report = exchange.submitOrder(order);

		expect(report.filledQuantity).toBeGreaterThan(0);
		expect(report.trades).toHaveLength(1);
	});

	it("random execution style stays deterministic for a fixed seed", () => {
		const exchange = new Exchange();

		const context = createObservableContext(exchange, {
			midPrice: 100,
			spread: 0,
		});
		const portfolioManager = new PortfolioManager();
		portfolioManager.createPortfolio("retail-5", 100000);
		portfolioManager.createPortfolio("retail-6", 100000);

		const agent1 = new RetailTraderAgent("retail-5", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 19,
			bias: "RANDOM",
			executionStyle: "RANDOM",
		});
		const agentContext1 = createAgentContext("retail-5", context, portfolioManager);

		const first = agent1.step(agentContext1);

		const agent2 = new RetailTraderAgent("retail-6", {
			referencePrice: 100,
			spread: 2,
			quantity: 8,
			seed: 19,
			bias: "RANDOM",
			executionStyle: "RANDOM",
		});
		const agentContext2 = createAgentContext("retail-6", context, portfolioManager);

		const second = agent2.step(agentContext2);

		expect(first[0].type).toBe("LIMIT");
		expect(second[0].type).toBe("LIMIT");

		if (
			first[0].type !== "LIMIT" ||
			second[0].type !== "LIMIT"
		) {
			throw new Error("Expected limit orders");
		}

		expect(first[0]).toMatchObject({
			quantity: second[0].quantity,
			side: second[0].side,
			price: second[0].price,
		});
	});

	it("random side bias can produce both buy and sell orders", () => {
		const exchange = new Exchange();

		const context = createObservableContext(exchange, {
			midPrice: 100,
		});

		const portfolioManager = new PortfolioManager();

		const sides = [1, 682].map((seed) => {
				portfolioManager.createPortfolio(`retail-${seed}`, 100000);
				const agentContext = createAgentContext(`retail-${seed}`, context, portfolioManager);
				return new RetailTraderAgent(`retail-${seed}`, {
					referencePrice: 100,
					spread: 2,
					quantity: 8,
					seed,
					bias: "RANDOM",
					executionStyle: "RANDOM",
				}).step(agentContext)[0].side;
			}
		);

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
		expect(simulator.getLimitedTradeHistory()).toHaveLength(0);
		expect(simulator.getEvents()).toHaveLength(0);

		expect(
			simulator.getParticpantPortfolios(),
		).toEqual([
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

	it("computes the correct observable market history", () => {
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

		let context = simulator.getObservableContext();

		expect(context.clock).toBe(0);
		expect(context.recentTrades).toHaveLength(0);
		expect(
			context.recentMidPriceSeries,
		).toHaveLength(0);

		simulator.runStep();

		context = simulator.getObservableContext();

		expect(context.clock).toBe(1);

		expect(context.orderBook.bids.length).toBeGreaterThan(0);
		expect(context.orderBook.asks.length).toBeGreaterThan(0);

		expect(context.midPrice).toBe(100);
		expect(context.spread).toBe(2);

		expect(context.recentTrades).toHaveLength(0);

		expect(context.recentMidPriceSeries).toEqual([100]);
	});

	it("maintains a bounded recent trade history", () => {
		const simulator = new Simulator({
			agents: buildDefaultAgents(123, 100),
			referencePrice: 100,
		});

		simulator.runSteps(10);

		const context = simulator.getObservableContext(
			5,
			5,
		);

		expect(
			context.recentTrades.length,
		).toBeLessThanOrEqual(5);

		expect(
			context.recentMidPriceSeries.length,
		).toBeLessThanOrEqual(5);
	});

	it("records one midprice observation per simulation step", () => {
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

		simulator.runSteps(5);

		const context = simulator.getObservableContext(
			20,
			20,
		);

		expect(
			context.recentMidPriceSeries,
		).toHaveLength(5);

		for (const price of context.recentMidPriceSeries) {
			expect(price).toBeGreaterThan(0);
		}
	});

	it("returns the most recent midprice observations", () => {
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

		simulator.runSteps(10);

		const context = simulator.getObservableContext(
			20,
			3,
		);

		expect(
			context.recentMidPriceSeries,
		).toHaveLength(3);

		expect(
			context.recentMidPriceSeries[
				context.recentMidPriceSeries.length - 1
			],
		).toBe(context.midPrice);
	});

	it("computes order imbalance from the current order book", () => {
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

		const context = simulator.getObservableContext();

		expect(
			context.orderImbalance.bidVolume,
		).toBe(10);

		expect(
			context.orderImbalance.askVolume,
		).toBe(10);

		expect(
			context.orderImbalance.imbalance,
		).toBe(0.5);
	});

	it("computes market history from the running simulation", () => {
		const simulator = new Simulator({
			agents: buildDefaultAgents(123, 100),
			referencePrice: 100,
		});

		let context = simulator.getObservableContext();

		expect(context.clock).toBe(0);
		expect(context.recentTrades).toHaveLength(0);
		expect(
			context.recentMidPriceSeries,
		).toHaveLength(0);

		simulator.runSteps(20);

		context = simulator.getObservableContext();

		expect(context.clock).toBe(20);

		expect(
			context.recentTrades.length,
		).toBeGreaterThan(0);

		expect(
			context.recentMidPriceSeries,
		).toHaveLength(20);

		expect(
			context.recentMidPriceSeries[0],
		).toBe(100);

		for (const level of context.orderBook.bids) {
			expect(level.price).toBeGreaterThan(0);
			expect(level.quantity).toBeGreaterThan(0);
		}

		for (const level of context.orderBook.asks) {
			expect(level.price).toBeGreaterThan(0);
			expect(level.quantity).toBeGreaterThan(0);
		}

		expect(context.spread).toBeGreaterThanOrEqual(0);

		const bidVolume =
			context.orderBook.bids.reduce(
				(sum, level) => sum + level.quantity,
				0,
			);

		const askVolume =
			context.orderBook.asks.reduce(
				(sum, level) => sum + level.quantity,
				0,
			);

		expect(
			context.orderImbalance.bidVolume,
		).toBe(bidVolume);

		expect(
			context.orderImbalance.askVolume,
		).toBe(askVolume);
	});

	it("respects the requested history limits", () => {
		const simulator = new Simulator({
			agents: buildDefaultAgents(123, 100),
			referencePrice: 100,
		});

		simulator.runSteps(50);

		const context = simulator.getObservableContext(
			5,
			10,
		);

		expect(
			context.recentTrades.length,
		).toBeLessThanOrEqual(5);

		expect(
			context.recentMidPriceSeries.length,
		).toBeLessThanOrEqual(10);

		expect(
			context.recentMidPriceSeries,
		).toHaveLength(10);
	});

	it("returns the most recent trades", () => {
		const simulator = new Simulator({
			agents: buildDefaultAgents(123, 100),
			referencePrice: 100,
		});

		simulator.runSteps(20);

		const fullHistory = simulator.getLimitedTradeHistory();

		const context = simulator.getObservableContext(
			5,
			20,
		);

		expect(context.recentTrades).toEqual(
			fullHistory.slice(-5),
		);
	});

	it("returns the most recent midprice observations", () => {
		const simulator = new Simulator({
			agents: buildDefaultAgents(123, 100),
			referencePrice: 100,
		});

		simulator.runSteps(20);

		const fullContext = simulator.getObservableContext(
			100,
			100,
		);

		const limitedContext = simulator.getObservableContext(
			100,
			5,
		);

		const expectedPrices =
			fullContext.recentMidPriceSeries.slice(-5);

		expect(
			limitedContext.recentMidPriceSeries,
		).toEqual(expectedPrices);
	});

	it("keeps market history separate from permanent trade history", () => {
		const simulator = new Simulator({
			agents: buildDefaultAgents(123, 100),
			referencePrice: 100,
		});

		simulator.runSteps(50);

		const fullTradeHistory =
			simulator.getLimitedTradeHistory();

		const context = simulator.getObservableContext(
			5,
			5,
		);

		expect(
			fullTradeHistory.length,
		).toBeGreaterThanOrEqual(
			context.recentTrades.length,
		);

		expect(context.recentTrades).toEqual(
			fullTradeHistory.slice(-5),
		);

		expect(
			context.recentMidPriceSeries,
		).toHaveLength(5);
	});

	it("resets market history when the simulator resets", () => {
		const simulator = new Simulator({
			agents: buildDefaultAgents(123, 100),
			referencePrice: 100,
		});

		simulator.runSteps(20);

		expect(
			simulator.getObservableContext()
				.recentMidPriceSeries.length,
		).toBeGreaterThan(0);

		expect(
			simulator.getLimitedTradeHistory().length,
		).toBeGreaterThan(0);

		simulator.reset();

		const context =
			simulator.getObservableContext();

		expect(context.clock).toBe(0);
		expect(context.recentTrades).toHaveLength(0);
		expect(
			context.recentMidPriceSeries,
		).toHaveLength(0);

		expect(
			simulator.getLimitedTradeHistory(),
		).toHaveLength(0);
	});
});