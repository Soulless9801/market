import { MatchingEngine } from "../matching";
import { OrderBook } from "../orderbook";
import type {
	CancelResult,
	ExecutionReport,
	IncomingOrder,
	NewOrderRequest,
	TradeEvent,
} from "../orders";

export class Exchange {
	private readonly orderBook = new OrderBook();
	private readonly matchingEngine = new MatchingEngine(this.orderBook, {
		createTradeId: () => this.nextTradeId(),
		now: () => this.nextEventTime(),
	});

	private readonly tradeHistory: TradeEvent[] = [];

	private orderSequence = 1;
	private tradeSequence = 1;
	private eventClock = 1;

	submitOrder(request: NewOrderRequest): ExecutionReport {
		const validation = this.validateOrder(request);
		const orderId = request.id ?? this.nextOrderId();

		if (!validation.valid) {
			return this.rejectedReport(orderId, validation.reason);
		}

		if (this.orderBook.hasOrder(orderId)) {
			return this.rejectedReport(
				orderId,
				`Order id "${orderId}" is already active in the order book`,
			);
		}


		const incomingOrder: IncomingOrder = {
			...request,
			id: orderId,
			timestamp: this.nextEventTime(),
		};

		const report = this.matchingEngine.execute(incomingOrder);
		this.tradeHistory.push(...report.trades);
		return report;
	}

	cancelOrder(orderId: string): CancelResult {
		return this.orderBook.cancel(orderId);
	}

	getOrderBookSnapshot(depth = 10) {
		return this.orderBook.getSnapshot(depth);
	}

	getTradeHistory(): TradeEvent[] {
		return [...this.tradeHistory];
	}

	private rejectedReport(
		orderId: string,
		reason: string,
	): ExecutionReport {
		return {
			orderId,
			status: "REJECTED",
			filledQuantity: 0,
			remainingQuantity: 0,
			wasAddedToBook: false,
			trades: [],
			rejectionReason: reason,
		};
	}

	private validateOrder(order: NewOrderRequest): {
		valid: boolean;
		reason: string;
	} {
		if (order.quantity <= 0 || !Number.isFinite(order.quantity)) {
			return {
				valid: false,
				reason: "Order quantity must be positive and finite",
			};
		}

		if (!Number.isInteger(order.quantity)) {
			return {
				valid: false,
				reason: "Order quantity must be an integer share count",
			};
		}

		if (order.type === "LIMIT") {
			if (order.price <= 0 || !Number.isFinite(order.price)) {
				return {
					valid: false,
					reason: "Limit price must be positive and finite",
				};
			}
		}

		return { valid: true, reason: "" };
	}

	private nextOrderId(): string {
		const orderId = `ord-${this.orderSequence}`;
		this.orderSequence += 1;
		return orderId;
	}

	private nextTradeId(): string {
		const tradeId = `trd-${this.tradeSequence}`;
		this.tradeSequence += 1;
		return tradeId;
	}

	private nextEventTime(): number {
		const now = this.eventClock;
		this.eventClock += 1;
		return now;
	}
}
