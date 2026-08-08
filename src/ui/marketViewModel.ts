import type { OrderBookSnapshot, TradeEvent } from "../engine";
import type { ParticipantStats } from "../simulation";

export interface BookRow {
	side: "BID" | "ASK";
	price: number;
	quantity: number;
	barWidth: number;
}

export interface TradeTapeEntry {
	id: string;
	timestamp: number;
	price: number;
	quantity: number;
	side: "BUY" | "SELL";
	participantId?: string;
}

export interface MarketViewModel {
	clock: number;
	midPrice: number;
	spread: number | null;
	tradeCount: number;
	totalVolume: number;
	imbalance: {
		bidVolume: number;
		askVolume: number;
		imbalance: number;
		bidPercent: number;
		askPercent: number;
	};
	participants: ParticipantSnapshot[];
	asks: BookRow[];
	bids: BookRow[];
	trades: TradeTapeEntry[];
	midPriceSeries: number[];
}

export interface ParticipantSnapshot {
	agentId: string;
	ordersSubmitted: number;
	inventory: number;
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

export function calculateOrderImbalance(snapshot: OrderBookSnapshot) {
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

export function buildTradeTape(trades: TradeEvent[], limit = 12): TradeTapeEntry[] {
	return trades.slice(-limit).reverse().map((trade) => ({
		id: trade.tradeId,
		timestamp: trade.timestamp,
		price: trade.price,
		quantity: trade.quantity,
		side: trade.aggressorSide,
	}));
}

export function buildParticipantSnapshots(stats: ParticipantStats[]): ParticipantSnapshot[] {
	return stats.map((stat) => ({
		agentId: stat.agentId,
		ordersSubmitted: stat.ordersSubmitted,
		inventory: stat.inventory,
	}));
}

export function buildBookRows(snapshot: OrderBookSnapshot): { bids: BookRow[]; asks: BookRow[] } {
	const bidLevels = [...snapshot.bids]
		.sort((left, right) => right.price - left.price)
		.map((level) => ({
			side: "BID" as const,
			price: level.price,
			quantity: level.quantity,
			barWidth: 0,
		}));
	const askLevels = [...snapshot.asks]
		.sort((left, right) => left.price - right.price)
		.map((level) => ({
			side: "ASK" as const,
			price: level.price,
			quantity: level.quantity,
			barWidth: 0,
		}));

	const maxQuantity = Math.max(
		1,
		...bidLevels.map((level) => level.quantity),
		...askLevels.map((level) => level.quantity),
	);

	return {
		bids: bidLevels.map((level) => ({
			...level,
			barWidth: Math.max(8, Math.round((level.quantity / maxQuantity) * 100)),
		})),
		asks: askLevels.map((level) => ({
			...level,
			barWidth: Math.max(8, Math.round((level.quantity / maxQuantity) * 100)),
		})),
	};
}

export function buildMarketViewModel(
	snapshot: OrderBookSnapshot,
	tradeHistory: TradeEvent[],
	participantStats: ParticipantStats[],
	clock: number,
	midPriceSeries: number[],
): MarketViewModel {
	const midPrice = calculateMidPrice(snapshot);
	const spread = calculateSpread(snapshot);
	const totalVolume = tradeHistory.reduce((sum, trade) => sum + trade.quantity, 0);
	const imbalance = calculateOrderImbalance(snapshot);
	const rows = buildBookRows(snapshot);

	return {
		clock,
		midPrice,
		spread,
		tradeCount: tradeHistory.length,
		totalVolume,
		imbalance,
		participants: buildParticipantSnapshots(participantStats),
		asks: rows.asks,
		bids: rows.bids,
		trades: buildTradeTape(tradeHistory),
		midPriceSeries,
	};
}
