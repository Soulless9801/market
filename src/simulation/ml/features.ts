import type { ObservableSimulatorContext } from '../simulator';

export function buildFeatures(context: ObservableSimulatorContext): number[] {
    return [
        (context.midPrice - context.recentMidPriceSeries[context.recentMidPriceSeries.length - 1]) / context.recentMidPriceSeries[context.recentMidPriceSeries.length - 1], // short return
        (context.midPrice - context.recentMidPriceSeries[0]) / context.recentMidPriceSeries[0], // long return
        context.spread / context.midPrice, // spread
        context.orderImbalance.imbalance, // imbalance
    ];
}
