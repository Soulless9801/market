import type {
	BookLevelSnapshot,
	CancelResult,
	OrderBookSnapshot,
	RestingOrder,
	Side,
} from "../orders";
import { OrderedStatisticTree } from "./OrderedStatisticTree";

interface SideState {
	levels: Map<number, PriceLevel>;
	prices: OrderedStatisticTree<number>;
}

interface PriceLevel {
	readonly price: number;
	head?: OrderNode;
	tail?: OrderNode;
	orderCount: number;
	totalQuantity: number;
}

interface OrderNode {
	readonly order: RestingOrder;
	readonly side: Side;
	readonly level: PriceLevel;
	prev?: OrderNode;
	next?: OrderNode;
}

export class OrderBook {
	private readonly bids: SideState = {
		levels: new Map<number, PriceLevel>(),
		prices: new OrderedStatisticTree<number>((a, b) => b - a),
	};

	private readonly asks: SideState = {
		levels: new Map<number, PriceLevel>(),
		prices: new OrderedStatisticTree<number>((a, b) => a - b),
	};

	private readonly orderIndex = new Map<string, OrderNode>();

	add(order: RestingOrder): void {
		const state = this.getSideState(order.side);
		let level = state.levels.get(order.price);
		if (!level) {
			level = {
				price: order.price,
				orderCount: 0,
				totalQuantity: 0,
			};
			state.levels.set(order.price, level);
			state.prices.insert(order.price);
		}

		const node: OrderNode = {
			order,
			side: order.side,
			level,
		};

		this.appendToLevel(level, node);
		this.orderIndex.set(order.id, node);
	}

	hasOrder(orderId: string): boolean {
		return this.orderIndex.has(orderId);
	}

	getBestPrice(side: Side): number | undefined {
		const state = this.getSideState(side);
		return state.prices.first();
	}

	getBestOrder(side: Side): RestingOrder | undefined {
		const bestPrice = this.getBestPrice(side);
		if (bestPrice === undefined) {
			return undefined;
		}

		const level = this.getSideState(side).levels.get(bestPrice);
		return level?.head?.order;
	}

	removeBestOrder(side: Side): RestingOrder | undefined {
		const bestNode = this.getBestNode(side);
		if (!bestNode) {
			return undefined;
		}

		return this.detachNode(bestNode);
	}

	fillBestOrder(side: Side, quantity: number): RestingOrder | undefined {
		if (quantity <= 0) {
			return undefined;
		}

		const bestNode = this.getBestNode(side);
		if (!bestNode) {
			return undefined;
		}

		if (quantity > bestNode.order.remainingQuantity) {
			throw new Error("Attempted to overfill best order");
		}

		bestNode.order.remainingQuantity -= quantity;
		bestNode.level.totalQuantity -= quantity;

		if (bestNode.order.remainingQuantity === 0) {
			this.detachNode(bestNode);
		}

		return bestNode.order;
	}

	cancel(orderId: string): CancelResult {
		const node = this.orderIndex.get(orderId);
		if (!node) {
			return {
				orderId,
				cancelled: false,
				cancelledQuantity: 0,
				reason: "Order not found in active book",
			};
		}

		const removedOrder = this.detachNode(node);

		return {
			orderId,
			cancelled: true,
			cancelledQuantity: removedOrder.remainingQuantity,
		};
	}

	getSnapshot(depth = 10): OrderBookSnapshot {
		return {
			bids: this.aggregateSide("BUY", depth),
			asks: this.aggregateSide("SELL", depth),
		};
	}

	private aggregateSide(side: Side, depth: number): BookLevelSnapshot[] {
		const state = this.getSideState(side);
		return state.prices.toArray(depth).map((price) => {
			const level = state.levels.get(price);
			if (!level) {
				throw new Error(
					`Missing level for price ${price}`,
				);
			}

			return {
				price,
				quantity: level.totalQuantity,
				orderCount: level.orderCount,
			};
		});
	}

	private getBestNode(side: Side): OrderNode | undefined {
		const bestPrice = this.getBestPrice(side);
		if (bestPrice === undefined) {
			return undefined;
		}

		const level = this.getSideState(side).levels.get(bestPrice);
		return level?.head;
	}

	private appendToLevel(level: PriceLevel, node: OrderNode): void {
		if (!level.head) {
			level.head = node;
			level.tail = node;
		} else {
			node.prev = level.tail;
			if (level.tail) {
				level.tail.next = node;
			}
			level.tail = node;
		}

		level.orderCount += 1;
		level.totalQuantity += node.order.remainingQuantity;
	}

	private detachNode(node: OrderNode): RestingOrder {
		const state = this.getSideState(node.side);
		const level = node.level;

		if (node.prev) {
			node.prev.next = node.next;
		} else {
			level.head = node.next;
		}

		if (node.next) {
			node.next.prev = node.prev;
		} else {
			level.tail = node.prev;
		}

		level.orderCount -= 1;
		level.totalQuantity -= node.order.remainingQuantity;
		this.orderIndex.delete(node.order.id);

		if (level.orderCount === 0) {
			state.levels.delete(level.price);
			state.prices.delete(level.price);
		}

		return node.order;
	}

	private getSideState(side: Side): SideState {
		return side === "BUY" ? this.bids : this.asks;
	}
}
