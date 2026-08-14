import type { NewOrderRequest, OrderBookSnapshot } from "../../engine";
import type { ObservableSimulatorContext } from "../simulator";

export type AgentSideBias = "BUY" | "SELL" | "RANDOM";
// Passive orders provide liquidity, while aggressive orders try to consume existing liquidity.
export type ExecutionStyle = "PASSIVE" | "AGGRESSIVE" | "RANDOM";

export interface TraderAgent {
	id: string;
	step(context: ObservableSimulatorContext): NewOrderRequest[];
}

export interface MarketMakerAgentOptions {
	referencePrice: number;
	spread: number;
	quantity: number;
}

export interface RetailTraderAgentOptions {
	referencePrice: number;
	spread: number;
	quantity: number;
	seed: number;
	maxPriceOffset?: number;
	bias?: AgentSideBias;
	executionStyle?: ExecutionStyle;
}

class SeededRandom {
	private state: number;

	constructor(seed: number) {
		this.state = seed >>> 0;
	}

	next(): number {
		this.state = (this.state * 1664525 + 1013904223) >>> 0;
		return this.state / 0x100000000;
	}
}

export function buildDefaultAgents(seed: number, referencePrice = 100): TraderAgent[] {
	return [
		new MarketMakerAgent("mm-1", {
			referencePrice,
			spread: 2,
			quantity: 10,
		}),
		new RetailTraderAgent("retail-1", {
			referencePrice,
			spread: 2,
			quantity: 8,
			seed,
			maxPriceOffset: 1,
			bias: "BUY",
			executionStyle: "AGGRESSIVE",
		}),
		new RetailTraderAgent("retail-2", {
			referencePrice,
			spread: 2,
			quantity: 8,
			seed: seed + 1,
			maxPriceOffset: 1,
			bias: "SELL",
			executionStyle: "AGGRESSIVE",
		}),
		new RetailTraderAgent("retail-3", {
			referencePrice,
			spread: 2,
			quantity: 8,
			seed: seed + 2,
			maxPriceOffset: 1,
			bias: "RANDOM",
			executionStyle: "RANDOM",
		}),
	];
}

export class MarketMakerAgent implements TraderAgent {
	readonly id: string;
	private readonly referencePrice: number;
	private readonly spread: number;
	private readonly quantity: number;

	constructor(id: string, options: MarketMakerAgentOptions) {
		this.id = id;
		this.referencePrice = options.referencePrice;
		this.spread = Math.max(1, options.spread);
		this.quantity = Math.max(1, options.quantity);
	}

	step(context: ObservableSimulatorContext): NewOrderRequest[] {
		const midPrice = context.midPrice ?? this.referencePrice;
		const halfSpread = this.spread / 2;
		const bidPrice = Math.max(1, midPrice - halfSpread);
		const askPrice = midPrice + halfSpread;

		return [
			{
				participantId: this.id,
				side: "BUY",
				type: "LIMIT",
				quantity: this.quantity,
				price: bidPrice,
			},
			{
				participantId: this.id,
				side: "SELL",
				type: "LIMIT",
				quantity: this.quantity,
				price: askPrice,
			},
		];
	}
}

export class RetailTraderAgent implements TraderAgent {
	readonly id: string;
	private readonly referencePrice: number;
	private readonly spread: number;
	private readonly quantity: number;
	private readonly maxPriceOffset: number;
	private readonly bias: AgentSideBias;
	private readonly executionStyle: ExecutionStyle;
	private readonly random: SeededRandom;

	constructor(id: string, options: RetailTraderAgentOptions) {
		this.id = id;
		this.referencePrice = options.referencePrice;
		this.spread = Math.max(1, options.spread);
		this.quantity = Math.max(1, options.quantity);
		this.maxPriceOffset = Math.max(0, options.maxPriceOffset ?? 1);
		this.bias = options.bias ?? "RANDOM";
		this.executionStyle = options.executionStyle ?? "RANDOM";
		this.random = new SeededRandom(options.seed);
	}

	step(context: ObservableSimulatorContext): NewOrderRequest[] {
		const midPrice = context.midPrice ?? this.referencePrice;
		const side = this.resolveSide();
		const executionStyle = this.resolveExecutionStyle();
		const price = this.calculateOrderPrice(side, executionStyle, midPrice, context.orderBook);
		const quantity = Math.max(1, Math.round(this.quantity + this.random.next() * 3));

		return [
			{
				participantId: this.id,
				side,
				type: "LIMIT",
				quantity,
				price,
			},
		];
	}

	private resolveSide(): "BUY" | "SELL" {
		if (this.bias === "BUY" || this.bias === "SELL") {
			return this.bias;
		}
		return this.random.next() < 0.5 ? "BUY" : "SELL";
	}

	private resolveExecutionStyle(): "PASSIVE" | "AGGRESSIVE" {
		if (this.executionStyle === "PASSIVE" || this.executionStyle === "AGGRESSIVE") {
			return this.executionStyle;
		}
		return this.random.next() < 0.5 ? "PASSIVE" : "AGGRESSIVE";
	}

	private calculateOrderPrice(side: "BUY" | "SELL", executionStyle: "PASSIVE" | "AGGRESSIVE", midPrice: number, snapshot: OrderBookSnapshot): number {
		const bestBid = snapshot.bids[0]?.price;
		const bestAsk = snapshot.asks[0]?.price;
		const halfSpread = this.spread / 2;
		const priceOffset = this.random.next() * this.maxPriceOffset;

		if (side === "BUY") {
			if (executionStyle === "PASSIVE") {
				if (bestBid !== undefined) {
					return Math.max(1, bestBid - priceOffset);
				}
				return Math.max(1, midPrice - halfSpread - priceOffset);
			}

			if (bestAsk !== undefined) {
				return bestAsk;
			}
			return Math.max(1, midPrice + halfSpread + priceOffset);
		}

		if (executionStyle === "PASSIVE") {
			if (bestAsk !== undefined) {
				return Math.max(1, bestAsk + priceOffset);
			}
			return Math.max(1, midPrice + halfSpread + priceOffset);
		}

		if (bestBid !== undefined) {
			return bestBid;
		}
		return Math.max(1, midPrice - halfSpread - priceOffset);
	}
}
