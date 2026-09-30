import { NamedRegistry } from "@/simulation/registry";
import { MLP_FEATURE_SCHEMA, CNN_FEATURE_SCHEMA } from "./feature-presets";
import type { ObservableSimulatorContext, ObservationRequirements } from '@/simulation/simulator';

export interface FeatureBuilder {
    // number of features produced by this builder
    featureCount: number;
    // build a feature vector from available info
    build(context: ObservableSimulatorContext): number[];
}

export class MLPFeatureBuilder implements FeatureBuilder {

    public readonly featureCount = 40;

    constructor() {}

    // helper function to compute log return safely
    private safeLogReturn(current: number, previous: number): number {
        return current > 0 && previous > 0
            ? Math.log(current / previous)
            : 0;
    }

    // helper function to ensure finite values or return zero
    private finiteOrZero(value: number): number {
        return Number.isFinite(value) ? value : 0;
    }

    //@override
    build(context: ObservableSimulatorContext): number[] {
        const prices = context.recentMidPriceSeries;
        const currentPrice = context.midPrice;
        const firstPrice = prices[0] ?? currentPrice;
        const lastPrice = prices[prices.length - 1] ?? currentPrice;
        const medianPrice = prices[Math.floor(prices.length / 2)] ?? currentPrice;
        const maxPrice = prices.length > 0 ? Math.max(...prices) : currentPrice;
        const minPrice = prices.length > 0 ? Math.min(...prices) : currentPrice;
        const referencePrice = context.referencePrice > 0
            ? context.referencePrice
            : 1;

        // compute log returns for recent prices
        const returns = context.recentMidPriceSeries.map((price, index, values) =>
            this.safeLogReturn(
                index === values.length - 1 ? context.midPrice : values[index + 1],
                price,
            ),
        );

        // std of returns
        const volatility = returns.length === 0
            ? 0
            : Math.sqrt(
                    returns.reduce((sum, value) => sum + value * value, 0) /
                    returns.length,
                );

        // sum up recent trade imbalance based on agressorside
        const recentTradeImbalance = context.recentTrades.reduce((sum, trade) => {
            return sum + (trade.aggressorSide === 'BUY' ? 1 : -1);
        }, 0) / Math.max(1, context.recentTrades.length);
        const bestBid = context.orderBook.bids[0];
        const bestAsk = context.orderBook.asks[0];
        const bidQuantity = bestBid?.quantity ?? 0;
        const askQuantity = bestAsk?.quantity ?? 0;
        const bookQuantity = bidQuantity + askQuantity;
        const topOfBookImbalance = bookQuantity > 0
            ? (bidQuantity - askQuantity) / bookQuantity
            : 0;
        const depthImbalances = Array.from({ length: 5 }, (_, index) => {
            const bidLevel = context.orderBook.bids[index]?.quantity ?? 0;
            const askLevel = context.orderBook.asks[index]?.quantity ?? 0;
            const levelQuantity = bidLevel + askLevel;
            return this.finiteOrZero(levelQuantity > 0
                ? (bidLevel - askLevel) / levelQuantity
                : 0);
        });

        const recentReturns = prices.map((price, index, values) =>
            this.finiteOrZero(this.safeLogReturn(
                index === values.length - 1 ? currentPrice : values[index + 1],
                price,
            )),
        );

        return [
            this.finiteOrZero(this.safeLogReturn(currentPrice, referencePrice)),
            this.finiteOrZero(this.safeLogReturn(maxPrice, referencePrice)),
            this.finiteOrZero(this.safeLogReturn(minPrice, referencePrice)),
            this.finiteOrZero(this.safeLogReturn(currentPrice, lastPrice)),
            this.finiteOrZero(this.safeLogReturn(currentPrice, medianPrice)),
            this.finiteOrZero(this.safeLogReturn(currentPrice, firstPrice)),
            this.finiteOrZero(currentPrice > 0 ? context.spread / currentPrice : 0),
            this.finiteOrZero(context.orderImbalance.imbalance),
            this.finiteOrZero(recentTradeImbalance),
            this.finiteOrZero(Math.log1p(volatility)),
            this.finiteOrZero(topOfBookImbalance),
            this.finiteOrZero(bestBid ? this.safeLogReturn(bestBid.price, currentPrice) : 0),
            this.finiteOrZero(bestAsk ? this.safeLogReturn(bestAsk.price, currentPrice) : 0),
            this.finiteOrZero(Math.log1p(bidQuantity)),
            this.finiteOrZero(Math.log1p(askQuantity)),
            ...recentReturns,
            ...depthImbalances,
        ];
    }
}

export class CNNFeatureBuilder implements FeatureBuilder {

    public readonly featureCount = 30;

    constructor() {}

    //@override
    build(context: ObservableSimulatorContext): number[] {
        const prices = context.recentMidPriceSeries.slice(-this.featureCount);
        // padding with current price if not enough data
        while (prices.length < this.featureCount) {
            prices.unshift(context.midPrice);
        }
        return [...prices];
    }
}

export interface FeatureDefinition extends ObservationRequirements {
	create(): FeatureBuilder;
	version: number;
	names: readonly string[];
	minimumWarmupSteps: number;
	historyOrder: string;
	padding: string;
	description: string;
}

/**
 * Layout definitions own feature order, shape, warmup and observation limits.
 * Dataset exporters and runtime agents ask this registry instead of inferring
 * those properties from a model name. A layout need not share an architecture ID.
 */
export class FeatureManager {
	private static readonly registry = new NamedRegistry<
		Readonly<FeatureDefinition>
	>("feature layout");
	static register(name: string, definition: FeatureDefinition): void {
		const counts = [
			definition.minimumWarmupSteps,
			definition.priceHistoryLimit,
			definition.tradeHistoryLimit,
		];
		if (
			counts.some(
				(value) =>
					!Number.isSafeInteger(value) ||
					value < 0,
			) ||
			!Number.isSafeInteger(definition.version) ||
			definition.version < 1 ||
			!Number.isSafeInteger(definition.bookDepth) ||
			definition.bookDepth < 1 ||
			definition.priceHistoryLimit > 1000 ||
			definition.tradeHistoryLimit > 1000 ||
			definition.names.length === 0 ||
			new Set(definition.names).size !==
				definition.names.length ||
			definition.names.some(
				(value) =>
					typeof value !== "string" ||
					!value.trim(),
			) ||
			[
				definition.historyOrder,
				definition.padding,
				definition.description,
			].some(
				(value) =>
					typeof value !== "string" ||
					!value.trim(),
			)
		) {
			throw new Error(`Invalid feature definition: ${name}`);
		}
		const builder = definition.create();
		if (builder.featureCount !== definition.names.length)
			throw new Error(
				`Feature count does not match metadata: ${name}`,
			);
		// Copy an allowlist: registration must not leak private fields to metadata.
		this.registry.register(
			name,
			Object.freeze({
				create: definition.create,
				version: definition.version,
				names: Object.freeze([...definition.names]),
				minimumWarmupSteps:
					definition.minimumWarmupSteps,
				priceHistoryLimit: definition.priceHistoryLimit,
				tradeHistoryLimit: definition.tradeHistoryLimit,
				bookDepth: definition.bookDepth,
				historyOrder: definition.historyOrder,
				padding: definition.padding,
				description: definition.description,
			}),
		);
	}
	static names(): string[] {
		return this.registry.names();
	}
	static describe(name: string): Readonly<FeatureDefinition> {
		return this.registry.get(name);
	}
	static create(name: string): FeatureBuilder {
		const definition = this.describe(name);
		const builder = definition.create();
		if (builder.featureCount !== definition.names.length)
			throw new Error(
				`Feature count does not match metadata: ${name}`,
			);
		return builder;
	}
}
FeatureManager.register("mlp", {
	...MLP_FEATURE_SCHEMA,
	create: () => new MLPFeatureBuilder(),
});
FeatureManager.register("cnn", {
	...CNN_FEATURE_SCHEMA,
	create: () => new CNNFeatureBuilder(),
});
