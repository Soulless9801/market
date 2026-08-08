import type { NewOrderRequest, OrderBookSnapshot } from "../../engine";

export type AgentSideBias = "BUY" | "SELL" | "RANDOM";

export interface SimulationContext {
	clock: number;
	snapshot: OrderBookSnapshot;
	midPrice: number;
	lastTradePrice?: number;
	tradeCount: number;
}

export interface TraderAgent {
	id: string;
	step(context: SimulationContext): NewOrderRequest[];
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

	step(context: SimulationContext): NewOrderRequest[] {
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
	private readonly random: SeededRandom;

	constructor(id: string, options: RetailTraderAgentOptions) {
		this.id = id;
		this.referencePrice = options.referencePrice;
		this.spread = Math.max(1, options.spread);
		this.quantity = Math.max(1, options.quantity);
		this.maxPriceOffset = Math.max(0, options.maxPriceOffset ?? 1);
		this.bias = options.bias ?? "RANDOM";
		this.random = new SeededRandom(options.seed);
	}

	step(context: SimulationContext): NewOrderRequest[] {
		const midPrice = context.midPrice ?? this.referencePrice;
		const side = this.resolveSide();
		const priceOffset = this.random.next() * this.maxPriceOffset;
		const price = side === "BUY"
			? Math.max(1, midPrice + priceOffset)
			: Math.max(1, midPrice + this.spread + priceOffset);
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
}
