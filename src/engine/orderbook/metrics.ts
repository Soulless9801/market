import type { OrderBookSnapshot } from "../orders";

export interface OrderImbalance {
	bidVolume: number;
	askVolume: number;
	imbalance: number;
	bidPercent: number;
	askPercent: number;
}

export function calculateMidPrice(snapshot: OrderBookSnapshot, fallback = 100): number {
    const bestBid = snapshot.bids[0]?.price;
    const bestAsk = snapshot.asks[0]?.price;
    if (bestBid !== undefined && bestAsk !== undefined) {
        return (bestBid + bestAsk) / 2;
    }
    return fallback;
}

export function calculateSpread(snapshot: OrderBookSnapshot): number | null {
    const bestBid = snapshot.bids[0]?.price;
    const bestAsk = snapshot.asks[0]?.price;
    if (bestBid === undefined || bestAsk === undefined) {
        return null;
    }
    return bestAsk - bestBid;
}

export function calculateRecentOrderImbalance(snapshot: OrderBookSnapshot): OrderImbalance {
    const bidVolume = snapshot.bids.reduce((sum, level) => sum + level.quantity, 0);
    const askVolume = snapshot.asks.reduce((sum, level) => sum + level.quantity, 0);
    const totalVolume = bidVolume + askVolume;
    const imbalance = totalVolume === 0 ? 0 : bidVolume / totalVolume;
    return {
        bidVolume,
        askVolume,
        imbalance,
        bidPercent: Math.round(imbalance * 1000) / 10,
        askPercent: Math.round((1 - imbalance) * 1000) / 10,
    };
}
