import { describe, expect, it } from "vitest";

import { Exchange } from "../engine";

describe("exchange", () => {
	it("matches crossing limit orders at resting order price and leaves residual quantity", () => {
		const exchange = new Exchange();

		let report = exchange.submitOrder({
			id: "buy-1",
			participantId: "participant-A",
			side: "BUY",
			type: "LIMIT",
			quantity: 100,
			price: 99,
		});

		expect(report.status).toBe("RESTING");
		expect(report.filledQuantity).toBe(0);
		expect(report.remainingQuantity).toBe(100);
		expect(report.wasAddedToBook).toBe(true);
		expect(report.trades).toHaveLength(0);

		report = exchange.submitOrder({
			id: "sell-1",
			participantId: "participant-B",
			side: "SELL",
			type: "LIMIT",
			quantity: 50,
			price: 98,
		});

		expect(report.status).toBe("FILLED");
		expect(report.trades).toHaveLength(1);
		expect(report.trades[0]).toMatchObject({
			buyOrderId: "buy-1",
			buyerParticipantId: "participant-A",
			sellOrderId: "sell-1",
			sellerParticipantId: "participant-B",
			makerOrderId: "buy-1",
			makerParticipantId: "participant-A",
			takerOrderId: "sell-1",
			takerParticipantId: "participant-B",
			quantity: 50,
			price: 99,
			aggressorSide: "SELL",
		});

		expect(exchange.getOrderBookSnapshot()).toEqual({
			bids: [{ price: 99, quantity: 50, orderCount: 1 }],
			asks: [],
		});
	});

	it("enforces price-time priority at a price level", () => {
		const exchange = new Exchange();

		exchange.submitOrder({
			id: "sell-early",
			participantId: "maker-1",
			side: "SELL",
			type: "LIMIT",
			quantity: 40,
			price: 101,
		});
		exchange.submitOrder({
			id: "sell-late",
			participantId: "maker-2",
			side: "SELL",
			type: "LIMIT",
			quantity: 40,
			price: 101,
		});

		const report = exchange.submitOrder({
			id: "buy-market",
			participantId: "taker-1",
			side: "BUY",
			type: "MARKET",
			quantity: 70,
		});

		expect(report.trades).toHaveLength(2);
		expect(report.trades[0]).toMatchObject({
			makerOrderId: "sell-early",
			quantity: 40,
		});
		expect(report.trades[1]).toMatchObject({
			makerOrderId: "sell-late",
			quantity: 30,
		});

		expect(exchange.getOrderBookSnapshot()).toEqual({
			bids: [],
			asks: [{ price: 101, quantity: 10, orderCount: 1 }],
		});
	});

	it("cancels a resting order and removes it from the order book", () => {
		const exchange = new Exchange();

		exchange.submitOrder({
			id: "buy-cancel",
			participantId: "participant-C",
			side: "BUY",
			type: "LIMIT",
			quantity: 25,
			price: 97,
		});

		const cancelResult = exchange.cancelOrder("buy-cancel");
		expect(cancelResult).toEqual({
			orderId: "buy-cancel",
			cancelled: true,
			cancelledQuantity: 25,
		});

		const report = exchange.submitOrder({
			id: "sell-1",
			participantId: "participant-D",
			side: "SELL",
			type: "LIMIT",
			quantity: 10,
			price: 96,
		});

		expect(report.status).toBe("RESTING");
		expect(report.filledQuantity).toBe(0);
		expect(report.remainingQuantity).toBe(10);
		expect(report.wasAddedToBook).toBe(true);
		expect(report.trades).toHaveLength(0);

		expect(exchange.getOrderBookSnapshot()).toEqual({
			bids: [],
			asks: [{ price: 96, quantity: 10, orderCount: 1 }],
		});
	});

	it("cancels middle queue order in O(1)-indexed path without breaking time priority", () => {
		const exchange = new Exchange();

		exchange.submitOrder({
			id: "sell-1",
			participantId: "maker-1",
			side: "SELL",
			type: "LIMIT",
			quantity: 10,
			price: 101,
		});
		exchange.submitOrder({
			id: "sell-2",
			participantId: "maker-2",
			side: "SELL",
			type: "LIMIT",
			quantity: 10,
			price: 101,
		});
		exchange.submitOrder({
			id: "sell-3",
			participantId: "maker-3",
			side: "SELL",
			type: "LIMIT",
			quantity: 10,
			price: 101,
		});

		expect(exchange.cancelOrder("sell-2").cancelled).toBe(true);

		const report = exchange.submitOrder({
			id: "buy-market-queue-check",
			participantId: "taker-2",
			side: "BUY",
			type: "MARKET",
			quantity: 20,
		});

		expect(report.trades).toHaveLength(2);
		expect(report.trades[0].makerOrderId).toBe("sell-1");
		expect(report.trades[1].makerOrderId).toBe("sell-3");

		expect(exchange.getOrderBookSnapshot()).toEqual({
			bids: [],
			asks: [],
		});
	});

	it("does not rest market orders when there is no available liquidity", () => {
		const exchange = new Exchange();

		let report = exchange.submitOrder({
			id: "buy-market-empty",
			participantId: "participant-D",
			side: "BUY",
			type: "MARKET",
			quantity: 10,
		});

		expect(report.status).toBe("UNFILLED");
		expect(report.filledQuantity).toBe(0);
		expect(report.remainingQuantity).toBe(10);
		expect(report.wasAddedToBook).toBe(false);
		expect(exchange.getOrderBookSnapshot()).toEqual({
			bids: [],
			asks: [],
		});

		exchange.submitOrder({
			id: "buy-limit",
			participantId: "participant-E",
			side: "BUY",
			type: "LIMIT",
			quantity: 5,
			price: 100,
		});

		exchange.submitOrder({
			id: "sell-limit",
			participantId: "participant-E",
			side: "SELL",
			type: "LIMIT",
			quantity: 5,
			price: 101,
		});

		report = exchange.submitOrder({
			id: "sell-market-empty",
			participantId: "participant-F",
			side: "SELL",
			type: "MARKET",
			quantity: 10,
		});

		expect(report.status).toBe("PARTIALLY_FILLED");
		expect(report.filledQuantity).toBe(5);
		expect(report.remainingQuantity).toBe(5);
		expect(report.wasAddedToBook).toBe(false);
		expect(exchange.getOrderBookSnapshot()).toEqual({
			bids: [],
			asks: [{ price: 101, quantity: 5, orderCount: 1 }],
		});
	});

	it("successfully rejects orders with invalid parameters", () => {
		const exchange = new Exchange();

		let report = exchange.submitOrder({
			id: "invalid-quantity",
			participantId: "participant-G",
			side: "BUY",
			type: "LIMIT",
			quantity: -10,
			price: 100,
		});

		expect(report.status).toBe("REJECTED");
		expect(report.rejectionReason).toBe(
			"Order quantity must be positive and finite",
		);

		report = exchange.submitOrder({
			id: "invalid-price",
			participantId: "participant-H",
			side: "SELL",
			type: "LIMIT",
			quantity: 10,
			price: -100,
		});

		expect(report.status).toBe("REJECTED");
		expect(report.rejectionReason).toBe(
			"Limit price must be positive and finite",
		);

		report = exchange.submitOrder({
			id: "invalid-quantity-type",
			participantId: "participant-I",
			side: "BUY",
			type: "LIMIT",
			quantity: 10.5,
			price: 100,
		});

		expect(report.status).toBe("REJECTED");
		expect(report.rejectionReason).toBe(
			"Order quantity must be an integer share count",
		);
	});
});
