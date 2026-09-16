import type { ObservableSimulatorContext } from '../simulator';

export const FEATURE_COUNT = 10;

export interface FeatureStatistics {
    means: number[];
    stds: number[];
}

export function buildFeatures(context: ObservableSimulatorContext): number[] {
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

    const returns = context.recentMidPriceSeries.map((price, index, values) =>
        safeLogReturn(
            index === values.length - 1 ? context.midPrice : values[index + 1],
            price,
        ),
    );
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
        finiteOrZero(safeLogReturn(currentPrice, referencePrice)),
        finiteOrZero(safeLogReturn(maxPrice, referencePrice)),
        finiteOrZero(safeLogReturn(minPrice, referencePrice)),
        finiteOrZero(safeLogReturn(currentPrice, lastPrice)),
        finiteOrZero(safeLogReturn(currentPrice, medianPrice)),
        finiteOrZero(safeLogReturn(currentPrice, firstPrice)),
        finiteOrZero(currentPrice > 0 ? context.spread / currentPrice : 0),
        finiteOrZero(context.orderImbalance.imbalance),
        finiteOrZero(recentTradeImbalance),
        finiteOrZero(Math.log1p(volatility)),
    ];
}

function safeLogReturn(current: number, previous: number): number {
    return current > 0 && previous > 0
        ? Math.log(current / previous)
        : 0;
}

function finiteOrZero(value: number): number {
    return Number.isFinite(value) ? value : 0;
}

export class FeatureNormalizer {
    readonly statistics: FeatureStatistics;

    constructor(statistics: FeatureStatistics) {
        if (
            statistics.means.length !== FEATURE_COUNT ||
            statistics.stds.length !== FEATURE_COUNT
        ) {
            throw new Error(`Expected ${FEATURE_COUNT} feature statistics.`);
        }

        this.statistics = {
            means: [...statistics.means],
            stds: [...statistics.stds],
        };
    }

    transform(features: number[]): number[] {
        if (features.length !== this.statistics.means.length) {
            throw new Error(`Expected ${this.statistics.means.length} features.`);
        }

        return features.map((value, index) =>
            (value - this.statistics.means[index]) /
                Math.max(this.statistics.stds[index], 1e-8),
        );
    }

    static fit(samples: number[][]): FeatureNormalizer {
        if (samples.length === 0 || samples.some((sample) => sample.length !== FEATURE_COUNT)) {
            throw new Error(`Expected non-empty samples with ${FEATURE_COUNT} features.`);
        }

        const means = Array.from({ length: FEATURE_COUNT }, (_, index) =>
            samples.reduce((sum, sample) => sum + sample[index], 0) / samples.length,
        );
        const stds = means.map((mean, index) =>
            Math.sqrt(
                samples.reduce((sum, sample) => sum + (sample[index] - mean) ** 2, 0) /
                samples.length,
            ),
        );

        return new FeatureNormalizer({ means, stds });
    }

    toJSON(): string {
        return JSON.stringify(this.statistics, null, 2);
    }

    static fromJSON(json: string): FeatureNormalizer {
        const statistics = JSON.parse(json) as FeatureStatistics;
        return new FeatureNormalizer(statistics);
    }
}
