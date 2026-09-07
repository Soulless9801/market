import type { ObservableSimulatorContext } from '../simulator';

export function buildFeatures(context: ObservableSimulatorContext): number[] {

    const n = context.recentMidPriceSeries.length;

    const lastMP = context.recentMidPriceSeries[n - 1];
    const firstMP = context.recentMidPriceSeries[0]; 
    const medMP = context.recentMidPriceSeries[Math.floor(n / 2)]; 

    const mxMP = Math.max(...context.recentMidPriceSeries);
    const mnMP = Math.min(...context.recentMidPriceSeries);

    // array of differences
    const diff = context.recentMidPriceSeries.map((_, i, arr) => ((i === n - 1 ? context.midPrice : arr[i + 1]) - arr[i]) / arr[i]);
    const diffStd = Math.sqrt(diff.reduce((sum, val) => sum + val * val, 0) / diff.length);

    // sum up recent trade imbalance based on agressorside
    const recentTradeImbalance = context.recentTrades.reduce((sum, trade) => {
        return sum + (trade.aggressorSide === 'BUY' ? 1 : -1);
    }, 0) / context.recentTrades.length;

    return [
        context.midPrice / context.referencePrice, // current mid price
        mxMP / context.referencePrice, // max mid price
        mnMP / context.referencePrice, // min mid price
        (context.midPrice - lastMP) / lastMP, // short return
        (context.midPrice - medMP) / medMP, // medium return
        (context.midPrice - firstMP) / firstMP, // long return
        context.spread / context.midPrice, // spread
        context.orderImbalance.imbalance, // imbalance
        recentTradeImbalance, // recent trade imbalance
        diffStd, // volatility
    ];
}
