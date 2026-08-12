import { OrderBook } from "../orderbook";
import type {
	ExecutionReport,
	IncomingOrder,
	RestingOrder,
	Side,
	TradeEvent,
} from "../orders";

interface MatchingEngineOptions {
	createTradeId: () => string;
	now: () => number;
}

export class MatchingEngine {
	private readonly orderBook: OrderBook;
	private readonly options: MatchingEngineOptions;

	constructor(orderBook: OrderBook, options: MatchingEngineOptions) {
		this.orderBook = orderBook;
		this.options = options;
	}

	execute(order: IncomingOrder): ExecutionReport {
		const oppositeSide = this.opposite(order.side);
		const trades: TradeEvent[] = [];
		let remainingQuantity = order.quantity;

		while (remainingQuantity > 0) {
			const bestOppositePrice =
				this.orderBook.getBestPrice(oppositeSide);
			if (bestOppositePrice === undefined) {
				break;
			}

			if (
				order.type === "LIMIT" &&
				!this.isCrossing(
					order.side,
					order.price ?? 0,
					bestOppositePrice,
				)
			) {
				break;
			}

			const maker = this.orderBook.getBestOrder(oppositeSide);
			if (!maker) {
				break;
			}

			const executionQuantity = Math.min(
				remainingQuantity,
				maker.remainingQuantity,
			);
			const filledMaker = this.orderBook.fillBestOrder(
				oppositeSide,
				executionQuantity,
			);
			if (!filledMaker) {
				throw new Error(
					"Failed to fill best order despite available liquidity",
				);
			}

			remainingQuantity -= executionQuantity;

			trades.push(
				this.buildTrade(
					order,
					filledMaker,
					executionQuantity,
				),
			);
		}

		if (order.type === "LIMIT" && remainingQuantity > 0) {
			this.orderBook.add(
				this.toRestingOrder(order, remainingQuantity),
			);
		}

		const filledQuantity = order.quantity - remainingQuantity;
		const wasAddedToBook =
			order.type === "LIMIT" && remainingQuantity > 0;

		return {
			orderId: order.id,
			status: this.resolveStatus(
				order.type,
				filledQuantity,
				remainingQuantity,
				wasAddedToBook,
			),
			filledQuantity,
			remainingQuantity,
			wasAddedToBook,
			trades,
		};
	}

	private buildTrade(
		order: IncomingOrder,
		maker: RestingOrder,
		quantity: number,
	): TradeEvent {

		const buyOrderId = order.side === "BUY" ? order.id : maker.id;
		const buyerParticipantId = order.side === "BUY" ? order.participantId : maker.participantId;
		const sellOrderId = order.side === "SELL" ? order.id : maker.id;
		const sellerParticipantId = order.side === "SELL" ? order.participantId : maker.participantId;

		return {
			tradeId: this.options.createTradeId(),
			price: maker.price, // On lit venues, fills occur at resting order (maker) price.
			quantity,
			buyOrderId,
			buyerParticipantId,
			sellOrderId,
			sellerParticipantId,
			makerOrderId: maker.id,
			makerParticipantId: maker.participantId,
			takerOrderId: order.id,
			takerParticipantId: order.participantId,
			aggressorSide: order.side,
			timestamp: this.options.now(),
		};
	}

	private toRestingOrder(
		order: IncomingOrder,
		remainingQuantity: number,
	): RestingOrder {
		return {
			id: order.id,
			participantId: order.participantId,
			side: order.side,
			price: order.price ?? 0,
			initialQuantity: order.quantity,
			remainingQuantity,
			timestamp: order.timestamp,
		};
	}

	private resolveStatus(
		type: IncomingOrder["type"],
		filledQuantity: number,
		remainingQuantity: number,
		wasAddedToBook: boolean,
	): ExecutionReport["status"] {
		if (wasAddedToBook) {
			return "RESTING";
		}

		if (remainingQuantity === 0) {
			return "FILLED";
		}

		if (type === "MARKET" && filledQuantity > 0) {
			return "PARTIALLY_FILLED";
		}

		return "UNFILLED";
	}

	private opposite(side: Side): Side {
		return side === "BUY" ? "SELL" : "BUY";
	}

	private isCrossing(
		side: Side,
		incomingPrice: number,
		restingPrice: number,
	): boolean {
		return side === "BUY"
			? incomingPrice >= restingPrice
			: incomingPrice <= restingPrice;
	}
}
