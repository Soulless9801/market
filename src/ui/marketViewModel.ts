import type { OrderBookSnapshot, TradeEvent } from "../engine";
import type { PortfolioSnapshot } from "../simulation/portfolio";
import {
	calculateMidPrice,
	calculateSpread,
	calculateOrderImbalance,
} from "../engine";

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
	cash: number;
	inventory: number;
	marketValue: number;
	equity: number;
	pnl: number;
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

export function buildParticipantSnapshots(stats: PortfolioSnapshot[]): ParticipantSnapshot[] {
	return stats.map((stat) => ({
		agentId: stat.participantId,
		ordersSubmitted: stat.ordersSubmitted,
		cash: stat.cash,
		marketValue: stat.marketValue,
		equity: stat.equity,
		inventory: stat.inventory,
		pnl: stat.pnl,
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
	portfolioSnapshots: PortfolioSnapshot[],
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
		participants: buildParticipantSnapshots(portfolioSnapshots),
		asks: rows.asks,
		bids: rows.bids,
		trades: buildTradeTape(tradeHistory),
		midPriceSeries,
	};
}
