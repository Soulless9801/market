import { describe, expect, it } from "vitest";

import { Exchange } from "../engine";
import { calculateMidPrice } from "../engine";
import { PortfolioManager } from "../simulation";

describe("PortfolioManager", () => {

	it("creates a portfolio with the given initial cash and zero inventory", () => {
		const manager = new PortfolioManager();

		const portfolio = manager.createPortfolio("trader-1", 50000);

		expect(portfolio).toEqual({
			participantId: "trader-1",
			initialCash: 50000,
			cash: 50000,
			inventory: 0,
		});
		expect(manager.getPortfolio("trader-1")).toEqual(portfolio);
	});

	it("throws when creating a portfolio for a participant that already exists", () => {
		const manager = new PortfolioManager();
		manager.createPortfolio("trader-1", 50000);

		expect(() => manager.createPortfolio("trader-1", 10000)).toThrow(
			"Portfolio for participant trader-1 already exists",
		);
	});

	it("returns undefined when getting a portfolio for a participant that was never created", () => {
		const manager = new PortfolioManager();

		expect(manager.getPortfolio("unknown")).toBeUndefined();
	});

	it("applies a trade by debiting the buyer, crediting the seller, and updating inventory on both sides", () => {
		const manager = new PortfolioManager();
		manager.createPortfolio("buyer-1", 10000);
		manager.createPortfolio("seller-1", 10000);

		const exchange = new Exchange();
		exchange.submitOrder({
			participantId: "seller-1",
			side: "SELL",
			type: "LIMIT",
			price: 100,
			quantity: 5,
		});
		const report = exchange.submitOrder({
			participantId: "buyer-1",
			side: "BUY",
			type: "LIMIT",
			price: 100,
			quantity: 5,
		});
		expect(report.trades).toHaveLength(1);

		manager.applyTrade(report.trades[0]);

		expect(manager.getPortfolio("buyer-1")).toMatchObject({ cash: 10000 - 100 * 5, inventory: 5 });
		expect(manager.getPortfolio("seller-1")).toMatchObject({ cash: 10000 + 100 * 5, inventory: -5 });
	});

	it("throws when applying a trade for a participant with no portfolio", () => {
		const manager = new PortfolioManager();
		manager.createPortfolio("buyer-1", 10000);
		// no portfolio created for "seller-1"

		const exchange = new Exchange();
		exchange.submitOrder({
			participantId: "seller-1",
			side: "SELL",
			type: "LIMIT",
			price: 100,
			quantity: 5,
		});
		const report = exchange.submitOrder({
			participantId: "buyer-1",
			side: "BUY",
			type: "LIMIT",
			price: 100,
			quantity: 5,
		});

		expect(() => manager.applyTrade(report.trades[0])).toThrow("Buyer or seller portfolio not found");
	});

	it("accumulates cash and inventory across multiple trades for the same participant", () => {
		const manager = new PortfolioManager();
		manager.createPortfolio("buyer-1", 10000);
		manager.createPortfolio("seller-1", 10000);

		const exchange = new Exchange();
		exchange.submitOrder({
			participantId: "seller-1",
			side: "SELL",
			type: "LIMIT",
			price: 100,
			quantity: 3,
		});
		const firstReport = exchange.submitOrder({
			participantId: "buyer-1",
			side: "BUY",
			type: "LIMIT",
			price: 100,
			quantity: 3,
		});
		manager.applyTrade(firstReport.trades[0]);

		exchange.submitOrder({
			participantId: "seller-1",
			side: "SELL",
			type: "LIMIT",
			price: 100,
			quantity: 2,
		});
		const secondReport = exchange.submitOrder({
			participantId: "buyer-1",
			side: "BUY",
			type: "LIMIT",
			price: 100,
			quantity: 2,
		});
		manager.applyTrade(secondReport.trades[0]);

		expect(manager.getPortfolio("buyer-1")).toMatchObject({ cash: 10000 - 100 * 5, inventory: 5 });
		expect(manager.getPortfolio("seller-1")).toMatchObject({ cash: 10000 + 100 * 5, inventory: -5 });
	});

	it("returns undefined when getting a snapshot for a participant that was never created", () => {
		const manager = new PortfolioManager();
		const exchange = new Exchange();

		expect(manager.getPortfolioSnapshot("unknown", exchange.getOrderBookSnapshot())).toBeUndefined();
	});

	it("computes marketValue, equity, and pnl from cash, inventory, and the current mid price", () => {
		const manager = new PortfolioManager();
        manager.createPortfolio("mm-1", 10000);
        manager.createPortfolio("seller-1", 10000);
		manager.createPortfolio("trader-1", 10000);

		const exchange = new Exchange();
		exchange.submitOrder({
			participantId: "mm-1",
			side: "BUY",
			type: "LIMIT",
			price: 99,
			quantity: 10,
		});
		exchange.submitOrder({
			participantId: "mm-1",
			side: "SELL",
			type: "LIMIT",
			price: 101,
			quantity: 10,
		});
		exchange.submitOrder({
			participantId: "seller-1",
			side: "SELL",
			type: "LIMIT",
			price: 100,
			quantity: 4,
		});
		const report = exchange.submitOrder({
			participantId: "trader-1",
			side: "BUY",
			type: "LIMIT",
			price: 100,
			quantity: 4,
		});
		expect(report.trades).toHaveLength(1);
		manager.applyTrade(report.trades[0]);

		const bookSnapshot = exchange.getOrderBookSnapshot();
		const midPrice = calculateMidPrice(bookSnapshot);
		const expectedCash = 10000 - 100 * 4;

		expect(manager.getPortfolioSnapshot("trader-1", bookSnapshot)).toEqual({
			participantId: "trader-1",
			cash: expectedCash,
			inventory: 4,
			midPrice,
			marketValue: 4 * midPrice,
			equity: expectedCash + 4 * midPrice,
			pnl: expectedCash + 4 * midPrice - 10000,
		});
	});

	it("returns zero market value and pnl equal to the cash change once inventory is flat", () => {
		const manager = new PortfolioManager();
        manager.createPortfolio("seller-1", 10000);
		manager.createPortfolio("trader-1", 10000);
        manager.createPortfolio("buyer-2", 10000);

		const exchange = new Exchange();
		exchange.submitOrder({
			participantId: "seller-1",
			side: "SELL",
			type: "LIMIT",
			price: 100,
			quantity: 5,
		});
		const buyReport = exchange.submitOrder({
			participantId: "trader-1",
			side: "BUY",
			type: "LIMIT",
			price: 100,
			quantity: 5,
		});
		expect(buyReport.trades).toHaveLength(1);
		manager.applyTrade(buyReport.trades[0]);

		exchange.submitOrder({
			participantId: "buyer-2",
			side: "BUY",
			type: "LIMIT",
			price: 100,
			quantity: 5,
		});
		const sellReport = exchange.submitOrder({
			participantId: "trader-1",
			side: "SELL",
			type: "LIMIT",
			price: 100,
			quantity: 5,
		});
		expect(sellReport.trades).toHaveLength(1);
		manager.applyTrade(sellReport.trades[0]);

		const snapshot = manager.getPortfolioSnapshot("trader-1", exchange.getOrderBookSnapshot());

		expect(snapshot).toMatchObject({
			cash: 10000,
			inventory: 0,
			marketValue: 0,
			equity: 10000,
			pnl: 0,
		});
	});

	it("clears all portfolios so previously created participants are no longer found", () => {
		const manager = new PortfolioManager();
		manager.createPortfolio("trader-1", 10000);

		manager.reset();

		expect(manager.getPortfolio("trader-1")).toBeUndefined();
	});

	it("allows recreating a portfolio for a participant after reset", () => {
		const manager = new PortfolioManager();
		manager.createPortfolio("trader-1", 10000);
		manager.reset();

		const portfolio = manager.createPortfolio("trader-1", 25000);

		expect(portfolio).toEqual({
			participantId: "trader-1",
			initialCash: 25000,
			cash: 25000,
			inventory: 0,
		});
	});
});