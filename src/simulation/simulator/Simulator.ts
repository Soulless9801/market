import { calculateRecentOrderImbalance, Exchange } from "../../engine";
import type { 
	ExecutionReport,
	OrderBookSnapshot, 
	OrderImbalance, 
	TradeEvent 
} from "../../engine";
import type { SimulationEvent } from "../events";
import type { TraderAgent } from "../agents";
import { PortfolioManager } from "../portfolio";
import type { PortfolioSnapshot } from "../portfolio";

export interface SimulatorOptions {
	exchange?: Exchange;
	agents: TraderAgent[];
	referencePrice?: number;
}

export interface StepResult {
	step: number;
	reports: ExecutionReport[];
	events: SimulationEvent[];
	trades: TradeEvent[];
}

// export interface SimulationContext {
// 	clock: number;
// 	snapshot: OrderBookSnapshot;
// 	midPrice: number;
// 	lastTradePrice?: number;
// 	tradeCount: number;
// }

export interface ObservableSimulatorContext {
	clock: number;
	midPrice: number;
	spread: number;
	orderBook: OrderBookSnapshot;
	recentTrades: TradeEvent[];
	recentMidPriceSeries: number[];
	orderImbalance: OrderImbalance;
}

export interface AgentSimulatorContext extends ObservableSimulatorContext {
	portfolio: PortfolioSnapshot;
}

export class Simulator {
	private exchange: Exchange;
	private readonly agents: TraderAgent[];
	private readonly referencePrice: number;
	private readonly midPriceHistory: number[] = [];
	private readonly events: SimulationEvent[] = [];
	// private readonly participantStats = new Map<string, ParticipantStats>();
	private readonly orderParticipants = new Map<string, string>();
	private clock = 0;

	private readonly portfolioManager = new PortfolioManager();

	constructor(options: SimulatorOptions) {
		this.exchange = options.exchange ?? new Exchange();
		this.agents = options.agents;
		this.referencePrice = options.referencePrice ?? 100;
		this.initializeParticipantPortfolios();
	}

	runStep(): StepResult {
		this.clock += 1;
		const observableContext = this.getObservableContext();
		this.midPriceHistory.push(observableContext.midPrice);
		const reports: ExecutionReport[] = [];
		const stepEvents: SimulationEvent[] = [];

		for (const agent of this.agents) {
			stepEvents.push({
				type: "agent-step",
				timestamp: this.clock,
				agentId: agent.id,
			});
			const agentContext = this.getAgentContext(agent.id, observableContext);

			const orders = agent.step(agentContext);
			for (const order of orders) {
				const report = this.exchange.submitOrder(order);
				this.portfolioManager.incrementOrdersSubmitted(agent.id);
				// this.recordOrderSubmission(agent.id, report.orderId);
				reports.push(report);
				stepEvents.push({
					type: "order-submitted",
					timestamp: this.clock,
					agentId: agent.id,
					orderId: report.orderId,
					report,
				});

				for (const trade of report.trades) {
					this.portfolioManager.applyTrade(trade);
				}

				if (report.trades.length > 0) {
					stepEvents.push({
						type: "trade",
						timestamp: this.clock,
						agentId: agent.id,
						orderId: report.orderId,
						tradeCount: report.trades.length,
					});
				}
			}
		}

		this.events.push(...stepEvents);
		return {
			step: this.clock,
			reports,
			events: stepEvents,
			trades: this.exchange.getTradeHistory(),
		};
	}

	runSteps(count: number): StepResult[] {
		const results: StepResult[] = [];
		for (let index = 0; index < count; index += 1) {
			results.push(this.runStep());
		}
		return results;
	}

	reset(): void {
		this.exchange = new Exchange();
		this.events.splice(0, this.events.length);
		this.portfolioManager.reset();
		this.midPriceHistory.splice(0, this.midPriceHistory.length);
		// this.participantStats.clear();
		this.orderParticipants.clear();
		this.clock = 0;
		this.initializeParticipantPortfolios();
	}

	getEvents(): SimulationEvent[] {
		return [...this.events];
	}

	getExchange(): Exchange {
		return this.exchange;
	}

	getOrderBookSnapshot(depth = 10): OrderBookSnapshot {
		return this.exchange.getOrderBookSnapshot(depth);
	}

	getTradeHistory(): TradeEvent[] {
		return this.exchange.getTradeHistory();
	}

	getMidPriceHistory(): number[] {
		return [...this.midPriceHistory];
	}

	getParticpantPortfolios(): PortfolioSnapshot[] {
		const snapshot = this.exchange.getOrderBookSnapshot();
		return this.agents.map((agent) => {
			const portfolioSnapshot = this.portfolioManager.getPortfolioSnapshot(agent.id, snapshot);
			if (!portfolioSnapshot) {
				throw new Error(`Portfolio snapshot for agent ${agent.id} not found`);
			}
			return portfolioSnapshot;
		});
	}

	getClock(): number {
		return this.clock;
	}

	private initializeParticipantPortfolios(): void {
		for (const agent of this.agents) {
			this.portfolioManager.createPortfolio(agent.id, 100000);
		}
	}

	// private createContext(): SimulationContext {
	// 	const snapshot = this.exchange.getOrderBookSnapshot();
	// 	const trades = this.exchange.getTradeHistory();
	// 	const lastTrade = trades[trades.length - 1];
	// 	const midPrice = this.calculateMidPrice(snapshot);
	// 	return {
	// 		clock: this.clock,
	// 		snapshot,
	// 		midPrice,
	// 		lastTradePrice: lastTrade?.price,
	// 		tradeCount: trades.length,
	// 	};
	// }

	getObservableContext(
		tradeHistoryLimit = 20,
		priceHistoryLimit = 20,
	): ObservableSimulatorContext {
		const orderBook = this.exchange.getOrderBookSnapshot();
		const tradeHistory = this.exchange.getTradeHistory();

		const midPrice = this.calculateMidPrice(orderBook);

		const bestBid = orderBook.bids[0]?.price;
		const bestAsk = orderBook.asks[0]?.price;

		const spread =
			bestBid !== undefined && bestAsk !== undefined
				? bestAsk - bestBid
				: 0;

		return {
			clock: this.clock,
			midPrice,
			spread,
			orderBook,
			recentTrades: tradeHistory.slice(-tradeHistoryLimit),
			recentMidPriceSeries:
				this.midPriceHistory.slice(-priceHistoryLimit),
			orderImbalance:
				calculateRecentOrderImbalance(orderBook),
		};
	}

	private calculateMidPrice(snapshot: OrderBookSnapshot): number {
		const bestBid = snapshot.bids[0]?.price;
		const bestAsk = snapshot.asks[0]?.price;
		if (bestBid !== undefined && bestAsk !== undefined) {
			return (bestBid + bestAsk) / 2;
		}
		return this.referencePrice;
	}

	getAgentContext(agentId: string, observableContext: ObservableSimulatorContext): AgentSimulatorContext {
		const portfolioSnapshot = this.portfolioManager.getPortfolioSnapshot(agentId, observableContext.orderBook);
		if (!portfolioSnapshot) {
			throw new Error(`Portfolio snapshot for agent ${agentId} not found`);
		}
		return {
			...observableContext,
			portfolio: portfolioSnapshot,
		};
	}
}
