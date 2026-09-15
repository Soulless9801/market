import { MatchingEngine } from "@/engine/matching";
import { OrderBook } from "@/engine/orderbook";
import type {
	CancelResult,
	ExecutionReport,
	IncomingOrder,
	NewOrderRequest,
	TradeEvent,
} from "@/engine/orders";
import { Deque } from "@/structs";

const MAX_TRADE_HISTORY = 1000;

export class Exchange {
	private readonly orderBook = new OrderBook();
	private readonly matchingEngine = new MatchingEngine(this.orderBook, {
		createTradeId: () => this.nextTradeId(),
		now: () => this.nextEventTime(),
	});

	private readonly tradeHistory: Deque<TradeEvent> = new Deque<TradeEvent>();

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
		for (const trade of report.trades) {
			this.tradeHistory.pushBack(trade);
		}
		this.clampTradeHistorySize();
		return report;
	}

	cancelOrder(orderId: string): CancelResult {
		return this.orderBook.cancel(orderId);
	}

	getOrderBookSnapshot(depth = 10) {
		return this.orderBook.getSnapshot(depth);
	}

	clampTradeHistorySize() {
		while (this.tradeHistory.size() > MAX_TRADE_HISTORY) {
			this.tradeHistory.popFront();
		}
	}

	getTradeHistory(): TradeEvent[] {
		return this.tradeHistory.toArray();
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
