import type { ObservableSimulatorContext } from '../simulator';

export interface FeatureBuilder {
    // build a feature vector from available info
    build(context: ObservableSimulatorContext): number[];
}

export class MLPFeatureBuilder implements FeatureBuilder {

    public static readonly featureCount = 10; // number of features produced by this builder

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
        }, 0) / context.recentTrades.length;

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
        ];
    }
}

export class CNNFeatureBuilder implements FeatureBuilder {

    public static readonly featureCount = 5; // number of features produced by this builder

    constructor() {}

    //@override
    build(context: ObservableSimulatorContext): number[] {
        const prices = context.recentMidPriceSeries.slice(-CNNFeatureBuilder.featureCount);
        // padding with current price if not enough data
        while (prices.length < CNNFeatureBuilder.featureCount) {
            prices.unshift(context.midPrice);
        }
        return [...prices];
    }
}

type FeatureBuilderConstructor = new () => FeatureBuilder;

export class FeatureManager {

    // map model name to feature builder constructor
    private static readonly registry = new Map<string, FeatureBuilderConstructor>([
        ['mlp', MLPFeatureBuilder],
        ['cnn', CNNFeatureBuilder],
    ]);

    // create a feature builder instance based on model name
    static create(modelName: string): FeatureBuilder {
        const builder = this.registry.get(modelName);
        if (!builder) throw new Error(`No feature builder registered for model: ${modelName}`);
        return new builder();
    }
}