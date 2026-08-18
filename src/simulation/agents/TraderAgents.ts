import type { NewOrderRequest, OrderBookSnapshot } from "../../engine";
import type { AgentSimulatorContext } from "../simulator";
import { MLP } from "../ml";

export type AgentSideBias = "BUY" | "SELL" | "RANDOM";
// Passive orders provide liquidity, while aggressive orders try to consume existing liquidity.
export type ExecutionStyle = "PASSIVE" | "AGGRESSIVE" | "RANDOM";

export interface TraderAgent {
	id: string;
	step(context: AgentSimulatorContext): NewOrderRequest[];
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

export interface MomentumTraderAgentOptions {
	referencePrice: number;
	spread: number;
	quantity: number;
	seed: number;
	lookback?: number;
	momentumThreshold?: number;
	executionStyle?: ExecutionStyle;
	maxPriceOffset?: number;
}

export interface MeanReversionTraderAgentOptions {
	referencePrice: number;
	spread: number;
	quantity: number;
	seed: number;
	lookback?: number;
	deviationThreshold?: number;
	executionStyle?: ExecutionStyle;
	maxPriceOffset?: number;
}

export interface ImbalanceTraderAgentOptions {
	referencePrice: number;
	spread: number;
	quantity: number;
	seed: number;
	buyThreshold?: number;
	sellThreshold?: number;
	executionStyle?: ExecutionStyle;
	maxPriceOffset?: number;
}

export type MLTraderAgentOptions = ImbalanceTraderAgentOptions; // placeholder for now, can be extended later

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
			executionStyle: "PASSIVE",
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
		new MomentumTraderAgent("momentum-1", {
			referencePrice,
			spread: 2,
			quantity: 5,
			seed: seed + 3,
			lookback: 5,
			momentumThreshold: 0.5,
			executionStyle: "AGGRESSIVE",
			maxPriceOffset: 1,
		}),
		new MomentumTraderAgent("momentum-2", {
			referencePrice,
			spread: 2,
			quantity: 5,
			seed: seed + 3,
			lookback: 5,
			momentumThreshold: 0.5,
			executionStyle: "PASSIVE",
			maxPriceOffset: 1,
		}),
		new MomentumTraderAgent("momentum-3", {
			referencePrice,
			spread: 2,
			quantity: 5,
			seed: seed + 3,
			lookback: 5,
			momentumThreshold: 0.5,
			executionStyle: "RANDOM",
			maxPriceOffset: 1,
		}),
		new MeanReversionTraderAgent("mean-reversion-1", {
			referencePrice,
			spread: 2,
			quantity: 5,
			seed: seed + 4,
			lookback: 5,
			deviationThreshold: 0.5,
			executionStyle: "AGGRESSIVE",
			maxPriceOffset: 1,
		}),
		new MeanReversionTraderAgent("mean-reversion-2", {
			referencePrice,
			spread: 2,
			quantity: 5,
			seed: seed + 4,
			lookback: 5,
			deviationThreshold: 0.5,
			executionStyle: "RANDOM",
			maxPriceOffset: 1,
		}),
		new MeanReversionTraderAgent("mean-reversion-3", {
			referencePrice,
			spread: 2,
			quantity: 5,
			seed: seed + 4,
			lookback: 5,
			deviationThreshold: 0.5,
			executionStyle: "PASSIVE",
			maxPriceOffset: 1,
		}),
		new ImbalanceTraderAgent("imbalance-1", {
			referencePrice,
			spread: 2,
			quantity: 5,
			seed: seed + 5,
			buyThreshold: 0.65,
			sellThreshold: 0.35,
			executionStyle: "AGGRESSIVE",
			maxPriceOffset: 1,
		}),
		new ImbalanceTraderAgent("imbalance-2", {
			referencePrice,
			spread: 2,
			quantity: 5,
			seed: seed + 5,
			buyThreshold: 0.65,
			sellThreshold: 0.35,
			executionStyle: "PASSIVE",
			maxPriceOffset: 1,
		}),
		new ImbalanceTraderAgent("imbalance-3", {
			referencePrice,
			spread: 2,
			quantity: 5,
			seed: seed + 5,
			buyThreshold: 0.65,
			sellThreshold: 0.35,
			executionStyle: "RANDOM",
			maxPriceOffset: 1,
		}),
		new MLTraderAgent("ml-1", {
			referencePrice,
			spread: 2,
			quantity: 5,
			seed: seed + 6,
			maxPriceOffset: 1,
		}),
	];
}

export function buildRandomAgents(seed: number, referencePrice = 100, count = 10): TraderAgent[] {
	const agents: TraderAgent[] = [];
	const random = new SeededRandom(seed);
	for (let i = 0; i < count; i++) {
		const agentSpin = random.next();
		if (agentSpin < 0.2) agents.push(randomMarketMakerAgent(`mm-${i}`, seed + i, referencePrice));
		else if (agentSpin < 0.5) agents.push(randomRetailTraderAgent(`retail-${i}`, seed + i, referencePrice));
		else if (agentSpin < 0.7) agents.push(randomMomentumTraderAgent(`momentum-${i}`, seed + i, referencePrice));
		else if (agentSpin < 0.9) agents.push(randomMeanReversionTraderAgent(`mean-reversion-${i}`, seed + i, referencePrice));
		else agents.push(randomImbalanceTraderAgent(`imbalance-${i}`, seed + i, referencePrice));
	}
	return agents;
}

export function buildAgents(seed: number, referencePrice = 100): TraderAgent[] {
	const agentCount = 10;
	const bots: TraderAgent[] = buildRandomAgents(seed, referencePrice, agentCount);
	const mlAgent = new MLTraderAgent("ml-1", {
		referencePrice,
		spread: 2,
		quantity: 5,
		seed: seed + agentCount,
		maxPriceOffset: 1,
	});
	return [...bots, mlAgent];
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

	step(context: AgentSimulatorContext): NewOrderRequest[] {
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

export function randomMarketMakerAgent(id: string, seed: number, referencePrice = 100): MarketMakerAgent {
	const random = new SeededRandom(seed);
	return new MarketMakerAgent(id, {
		referencePrice,
		spread: random.next() * 2 + 1,
		quantity: Math.floor(random.next() * 10) + 1,
	});
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

	step(context: AgentSimulatorContext): NewOrderRequest[] {
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
		if (this.executionStyle !== "RANDOM") return this.executionStyle;
		return this.random.next() < 0.5 ? "PASSIVE" : "AGGRESSIVE";
	}

	private calculateOrderPrice(side: "BUY" | "SELL", executionStyle: "PASSIVE" | "AGGRESSIVE", midPrice: number, snapshot: OrderBookSnapshot): number {
		const bestBid = snapshot.bids[0]?.price;
		const bestAsk = snapshot.asks[0]?.price;
		const halfSpread = this.spread / 2;
		const priceOffset = this.random.next() * this.maxPriceOffset;

		if (side === "BUY") {
			if (executionStyle === "PASSIVE") {
				if (bestBid !== undefined) { // passive buy below best bid
					return Math.max(1, bestBid - priceOffset);
				}
				return Math.max(1, midPrice - halfSpread - priceOffset);
			}

			if (bestAsk !== undefined) { // aggressive buy at best ask
				return bestAsk;
			}
			return Math.max(1, midPrice + halfSpread + priceOffset);
		}

		if (executionStyle === "PASSIVE") {
			if (bestAsk !== undefined) { // passive ask above best ask
				return Math.max(1, bestAsk + priceOffset);
			}
			return Math.max(1, midPrice + halfSpread + priceOffset);
		}

		if (bestBid !== undefined) { // aggressive sell at best bid
			return bestBid;
		}
		return Math.max(1, midPrice - halfSpread - priceOffset);
	}
}

export function randomRetailTraderAgent(id: string, seed: number, referencePrice = 100): RetailTraderAgent {
	const random = new SeededRandom(seed);
	const biasSpin = random.next();
	const bias: AgentSideBias = biasSpin < 0.4 ? "BUY" : biasSpin < 0.8 ? "SELL" : "RANDOM";
	const executionStyleSpin = random.next();
	let executionStyle: ExecutionStyle = "RANDOM";
	if (bias === "BUY") executionStyle = executionStyleSpin < 0.7 ? "AGGRESSIVE" : "PASSIVE";
	if (bias === "SELL") executionStyle = executionStyleSpin < 0.7 ? "PASSIVE" : "AGGRESSIVE";
	return new RetailTraderAgent(id, {
		referencePrice,
		bias,
		executionStyle,
		spread: random.next() * 2 + 1,
		quantity: Math.floor(random.next() * 10) + 1,
		maxPriceOffset: random.next() * 1 + 0.5,
		seed,
	});
}

export class MomentumTraderAgent implements TraderAgent {
	readonly id: string;

	// private readonly referencePrice: number;
	private readonly spread: number;
	private readonly quantity: number;
	private readonly lookback: number;
	private readonly momentumThreshold: number;
	private readonly executionStyle: ExecutionStyle;
	private readonly maxPriceOffset: number;
	private readonly random: SeededRandom;

	constructor(
		id: string,
		options: MomentumTraderAgentOptions,
	) {
		this.id = id;
		// this.referencePrice = options.referencePrice;
		this.spread = Math.max(1, options.spread);
		this.quantity = Math.max(1, options.quantity);
		this.lookback = Math.max(1, options.lookback ?? 5);
		this.momentumThreshold =
			Math.max(0, options.momentumThreshold ?? 0);
		this.executionStyle =
			options.executionStyle ?? "AGGRESSIVE";
		this.maxPriceOffset =
			Math.max(0, options.maxPriceOffset ?? 1);

		this.random = new SeededRandom(options.seed);
	}

	step(context: AgentSimulatorContext): NewOrderRequest[] {
		const side = this.resolveSide(context);
		const price = this.calculateOrderPrice(
			side,
			context,
		);

		const quantity = Math.max(
			1,
			Math.round(
				this.quantity +
					this.random.next() * 3,
			),
		);

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

	private resolveSide(
		context: AgentSimulatorContext,
	): "BUY" | "SELL" {
		const history = context.recentMidPriceSeries;

		if (history.length < 2) {
			return "BUY";
		}

		const lookbackStart = Math.max(
			0,
			history.length - this.lookback,
		);

		const reference =
			history[lookbackStart];

		const momentum =
			context.midPrice - reference;

		if (momentum > this.momentumThreshold) {
			return "BUY";
		}

		if (momentum < -this.momentumThreshold) {
			return "SELL";
		}

		return this.random.next() < 0.5
			? "BUY"
			: "SELL";
	}

	private calculateOrderPrice(
		side: "BUY" | "SELL",
		context: AgentSimulatorContext,
	): number {
		const bestBid =
			context.orderBook.bids[0]?.price;

		const bestAsk =
			context.orderBook.asks[0]?.price;

		const offset =
			this.random.next() *
			this.maxPriceOffset;

		if (this.executionStyle === "PASSIVE") {
			if (side === "BUY") {
				return Math.max(
					1,
					(bestBid ??
						context.midPrice -
							this.spread / 2) -
						offset,
				);
			}

			return Math.max(
				1,
				(bestAsk ??
					context.midPrice +
						this.spread / 2) +
					offset,
			);
		}

		if (side === "BUY") {
			return (
				bestAsk ??
				context.midPrice +
					this.spread / 2 +
					offset
			);
		}

		return Math.max(
			1,
			bestBid ??
				context.midPrice -
					this.spread / 2 -
					offset,
		);
	}
}

export function randomMomentumTraderAgent(id: string, seed: number, referencePrice = 100): MomentumTraderAgent {
	const random = new SeededRandom(seed);
	return new MomentumTraderAgent(id, {
		referencePrice,
		spread: random.next() * 2 + 1,
		quantity: Math.floor(random.next() * 10) + 1,
		lookback: 10,
		momentumThreshold: random.next() * 1 + 0.5,
		executionStyle: random.next() < 0.5 ? "AGGRESSIVE" : "PASSIVE",
		maxPriceOffset: random.next() * 1 + 0.5,
		seed,
	});
}

export class MeanReversionTraderAgent
	implements TraderAgent
{
	readonly id: string;

	// private readonly referencePrice: number;
	private readonly spread: number;
	private readonly quantity: number;
	private readonly lookback: number;
	private readonly deviationThreshold: number;
	private readonly executionStyle: ExecutionStyle;
	private readonly maxPriceOffset: number;
	private readonly random: SeededRandom;

	constructor(
		id: string,
		options: MeanReversionTraderAgentOptions,
	) {
		this.id = id;
		// this.referencePrice = options.referencePrice;
		this.spread = Math.max(1, options.spread);
		this.quantity = Math.max(1, options.quantity);
		this.lookback = Math.max(1, options.lookback ?? 5);
		this.deviationThreshold =
			Math.max(
				0,
				options.deviationThreshold ?? 0.5,
			);

		this.executionStyle =
			options.executionStyle ?? "PASSIVE";

		this.maxPriceOffset =
			Math.max(
				0,
				options.maxPriceOffset ?? 1,
			);

		this.random = new SeededRandom(options.seed);
	}

	step(
		context: AgentSimulatorContext,
	): NewOrderRequest[] {
		const side = this.resolveSide(context);

		const price =
			this.calculateOrderPrice(
				side,
				context,
			);

		const quantity = Math.max(
			1,
			Math.round(
				this.quantity +
					this.random.next() * 3,
			),
		);

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

	private resolveSide(
		context: AgentSimulatorContext,
	): "BUY" | "SELL" {
		const history =
			context.recentMidPriceSeries;

		if (history.length === 0) {
			return "BUY";
		}

		const window = history.slice(-this.lookback);

		const mean =
			window.reduce((sum, price) => sum + price, 0) /
			window.length;

		const deviation = context.midPrice - mean;

		if (
			deviation >
			this.deviationThreshold
		) {
			return "SELL";
		}

		if (
			deviation <
			-this.deviationThreshold
		) {
			return "BUY";
		}

		return this.random.next() < 0.5
			? "BUY"
			: "SELL";
	}

	private calculateOrderPrice(
		side: "BUY" | "SELL",
		context: AgentSimulatorContext,
	): number {
		const bestBid =
			context.orderBook.bids[0]?.price;

		const bestAsk =
			context.orderBook.asks[0]?.price;

		const offset =
			this.random.next() *
			this.maxPriceOffset;

		if (
			this.executionStyle ===
			"PASSIVE"
		) {
			if (side === "BUY") {
				return Math.max(
					1,
					(bestBid ??
						context.midPrice -
							this.spread / 2) -
						offset,
				);
			}

			return Math.max(
				1,
				(bestAsk ??
					context.midPrice +
						this.spread / 2) +
					offset,
			);
		}

		if (side === "BUY") {
			return (
				bestAsk ??
				context.midPrice +
					this.spread / 2 +
					offset
			);
		}

		return Math.max(
			1,
			bestBid ??
				context.midPrice -
					this.spread / 2 -
					offset,
		);
	}
}

export function randomMeanReversionTraderAgent(id: string, seed: number, referencePrice = 100): MeanReversionTraderAgent {
	const random = new SeededRandom(seed);
	return new MeanReversionTraderAgent(id, {
		referencePrice,
		spread: random.next() * 2 + 1,
		quantity: Math.floor(random.next() * 10) + 1,
		lookback: 10,
		deviationThreshold: random.next() * 1 + 0.5,
		executionStyle: random.next() < 0.5 ? "PASSIVE" : "AGGRESSIVE",
		maxPriceOffset: random.next() * 1 + 0.5,
		seed,
	});
}

export class ImbalanceTraderAgent
	implements TraderAgent
{
	readonly id: string;

	// private readonly referencePrice: number;
	private readonly spread: number;
	private readonly quantity: number;
	private readonly buyThreshold: number;
	private readonly sellThreshold: number;
	private readonly executionStyle: ExecutionStyle;
	private readonly maxPriceOffset: number;
	private readonly random: SeededRandom;

	constructor(
		id: string,
		options: ImbalanceTraderAgentOptions,
	) {
		this.id = id;
		// this.referencePrice =
		// 	options.referencePrice;

		this.spread =
			Math.max(1, options.spread);

		this.quantity =
			Math.max(1, options.quantity);

		this.buyThreshold =
			options.buyThreshold ?? 0.65;

		this.sellThreshold =
			options.sellThreshold ?? 0.35;

		this.executionStyle =
			options.executionStyle ?? "AGGRESSIVE";

		this.maxPriceOffset =
			options.maxPriceOffset ?? 1;

		this.random =
			new SeededRandom(options.seed);
	}

	step(
		context: AgentSimulatorContext,
	): NewOrderRequest[] {
		const imbalance =
			context.orderImbalance
				.imbalance;

		let side: "BUY" | "SELL";

		if (imbalance >= this.buyThreshold) {
			side = "BUY";
		} else if (
			imbalance <= this.sellThreshold
		) {
			side = "SELL";
		} else {
			side =
				this.random.next() < 0.5
					? "BUY"
					: "SELL";
		}

		const price =
			this.calculateOrderPrice(
				side,
				context,
			);

		const quantity = Math.max(
			1,
			Math.round(
				this.quantity +
					this.random.next() * 3,
			),
		);

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

	private calculateOrderPrice(
		side: "BUY" | "SELL",
		context: AgentSimulatorContext,
	): number {
		const bestBid =
			context.orderBook.bids[0]?.price;

		const bestAsk =
			context.orderBook.asks[0]?.price;

		const offset =
			this.random.next() *
			this.maxPriceOffset;

		if (
			this.executionStyle ===
			"PASSIVE"
		) {
			if (side === "BUY") {
				return Math.max(
					1,
					(bestBid ??
						context.midPrice -
							this.spread / 2) -
						offset,
				);
			}

			return Math.max(
				1,
				(bestAsk ??
					context.midPrice +
						this.spread / 2) +
					offset,
			);
		}

		if (side === "BUY") {
			return (
				bestAsk ??
				context.midPrice +
					this.spread / 2 +
					offset
			);
		}

		return Math.max(
			1,
			bestBid ??
				context.midPrice -
					this.spread / 2 -
					offset,
		);
	}
}

export function randomImbalanceTraderAgent(id: string, seed: number, referencePrice = 100): ImbalanceTraderAgent {
	const random = new SeededRandom(seed);
	return new ImbalanceTraderAgent(id, {
		referencePrice,
		spread: random.next() * 2 + 1,
		quantity: Math.floor(random.next() * 10) + 1,
		buyThreshold: random.next() * 0.2 + 0.6, // between 0.6 and 0.9
		sellThreshold: random.next() * 0.2 + 0.2, // between 0.1 and 0.4
		executionStyle: random.next() < 0.5 ? "AGGRESSIVE" : "PASSIVE",
		maxPriceOffset: random.next() * 1 + 0.5,
		seed,
	});
}

export class MLTraderAgent implements TraderAgent {
	readonly id: string;
	// private readonly referencePrice: number;
	private readonly spread: number;
	private readonly quantity: number;
	private readonly executionStyle: ExecutionStyle;
	private readonly maxPriceOffset: number;
	private readonly random: SeededRandom;

	constructor(id: string, options: MLTraderAgentOptions) {
		this.id = id;
		// this.referencePrice = options.referencePrice;
		this.spread = Math.max(1, options.spread);
		this.quantity = Math.max(1, options.quantity);
		this.executionStyle = options.executionStyle ?? "AGGRESSIVE";
		this.maxPriceOffset = Math.max(0, options.maxPriceOffset ?? 1);
		this.random = new SeededRandom(options.seed);
	}

	step(context: AgentSimulatorContext): NewOrderRequest[] {
		const side = this.resolveSide(context);
		const price = this.calculateOrderPrice(side, context);
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

	private resolveSide(context: AgentSimulatorContext): "BUY" | "SELL" {
		const model = new MLP();
		const input = [
			context.recentMidPriceSeries[context.recentMidPriceSeries.length - 1] - context.recentMidPriceSeries[context.recentMidPriceSeries.length - 2], // short return
			context.recentMidPriceSeries[context.recentMidPriceSeries.length - 1] - context.recentMidPriceSeries[0], // long return
			context.spread, // spread
			context.orderImbalance.imbalance, // imbalance
		];
		const output = model.predict(input);
		return output[0] > output[1] ? "BUY" : "SELL";
	}

	private calculateOrderPrice(side: "BUY" | "SELL", context: AgentSimulatorContext): number {
		const bestBid = context.orderBook.bids[0]?.price;
		const bestAsk = context.orderBook.asks[0]?.price;
		const offset = this.random.next() * this.maxPriceOffset;

		if (this.executionStyle === "PASSIVE") {
			if (side === "BUY") {
				return Math.max(1, (bestBid ?? context.midPrice - this.spread / 2) - offset);
			}
			return Math.max(1, (bestAsk ?? context.midPrice + this.spread / 2) + offset);
		}

		if (side === "BUY") {
			return bestAsk ?? context.midPrice + this.spread / 2 + offset;
		}

		return Math.max(1, bestBid ?? context.midPrice - this.spread / 2 - offset);
	}
}