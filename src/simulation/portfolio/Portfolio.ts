import { calculateMidPrice } from "../../engine/orderbook";
import type { TradeEvent } from "../../engine/orders";
import type { OrderBookSnapshot } from "../../engine/orders";

export interface Portfolio {
    participantId: string;
    initialCash: number;
    cash: number;
    inventory: number; // single market for now
}

export interface PortfolioSnapshot {
    participantId: string;
    cash: number;
    inventory: number;
    midPrice: number;
    marketValue: number;
    equity: number;
    pnl: number;
}

export class PortfolioManager {
    private readonly portfolios: Map<string, Portfolio> = new Map();

    constructor() {}

    createPortfolio(participantId: string, initialCash: number): Portfolio {
        if (this.portfolios.has(participantId)) {
            throw new Error(`Portfolio for participant ${participantId} already exists`);
        }
        const portfolio: Portfolio = {
            participantId,
            initialCash,
            cash: initialCash,
            inventory: 0,
        };
        this.portfolios.set(participantId, portfolio);
        return portfolio;
    }

    getPortfolio(participantId: string): Portfolio | undefined {
        return this.portfolios.get(participantId);
    }

    applyTrade(trade: TradeEvent): void {
        const buyerPortfolio = this.portfolios.get(trade.buyerParticipantId);
        const sellerPortfolio = this.portfolios.get(trade.sellerParticipantId);

        if (!buyerPortfolio || !sellerPortfolio) {
            throw new Error("Buyer or seller portfolio not found");
        }

        // Update buyer's portfolio
        buyerPortfolio.cash -= trade.price * trade.quantity;
        buyerPortfolio.inventory += trade.quantity;

        // Update seller's portfolio
        sellerPortfolio.cash += trade.price * trade.quantity;
        sellerPortfolio.inventory -= trade.quantity;
    }

    getPortfolioSnapshot(participantId: string, orderBookSnapshot: OrderBookSnapshot): PortfolioSnapshot | undefined {
        const portfolio = this.portfolios.get(participantId);
        if (!portfolio) {
            return undefined;
        }
        
        const midPrice = calculateMidPrice(orderBookSnapshot);
        const marketValue = portfolio.inventory * midPrice;
        const equity = portfolio.cash + marketValue;
        const pnl = equity - portfolio.initialCash;

        return {
            participantId: portfolio.participantId,
            cash: portfolio.cash,
            inventory: portfolio.inventory,
            midPrice: midPrice,
            marketValue: marketValue,
            equity: equity,
            pnl: pnl,
        };
    }

    reset(): void {
        this.portfolios.clear();
    }
}