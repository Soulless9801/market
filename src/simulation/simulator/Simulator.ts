import type { 
	ExecutionReport,
	NewOrderRequest,
	LimitedTradeEvent,
	OrderBookSnapshot, 
	OrderImbalance, 
	TradeEvent 
} from "@/engine";
import { calculateRecentOrderImbalance, Exchange } from "@/engine";
import type { SimulationEvent, TraderAgent, PortfolioSnapshot } from "@/simulation";
import { PortfolioManager } from "@/simulation";
import { Deque } from "@/structs";

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

// aggregate simulator statistics
export interface SimulatorStatistics {
	volume: number;
	tradeCount: number;
}

// export interface SimulationContext {
// 	clock: number;
// 	snapshot: OrderBookSnapshot;
// 	midPrice: number;
// 	lastTradePrice?: number;
// 	tradeCount: number;
// }

/** Public observation requested by an agent or feature layout. */
export interface ObservationRequirements {
	tradeHistoryLimit: number;
	priceHistoryLimit: number;
	bookDepth: number;
}

export interface ObservableSimulatorContext {
	clock: number;
	midPrice: number;
	referencePrice: number;
	spread: number;
	orderBook: OrderBookSnapshot;
	recentTrades: LimitedTradeEvent[];
	recentMidPriceSeries: number[];
	orderImbalance: OrderImbalance;
}

export interface AgentSimulatorContext extends ObservableSimulatorContext {
	portfolio: PortfolioSnapshot;
	/** Completed step represented by the observation; clock remains the agent's decision step. */
	observationClock?: number;
}

const MAX_MID_PRICE_HISTORY = 1000;
const MAX_EVENT_HISTORY = 1000;

export class Simulator {
	private exchange: Exchange;
	private readonly agents: TraderAgent[];
	private readonly referencePrice: number;
	private readonly midPriceHistory: Deque<number> = new Deque<number>();
	private readonly events: Deque<SimulationEvent> = new Deque<SimulationEvent>();
	// private readonly participantStats = new Map<string, ParticipantStats>();
	private readonly orderParticipants = new Map<string, string>();
	private clock = 0;
	private stepping = false;
	private faulted = false;

	private readonly portfolioManager = new PortfolioManager();
	private statistics: SimulatorStatistics;

	private initializeParticipantPortfolios(): void {
		for (const agent of this.agents) {
			this.portfolioManager.createPortfolio(agent.id, 100000);
		}
	}

	constructor(options: SimulatorOptions) {
		this.exchange = options.exchange ?? new Exchange();
		this.agents = [...options.agents];
		this.referencePrice = options.referencePrice ?? 100;
		this.initializeParticipantPortfolios();
		this.statistics = {
			volume: 0,
			tradeCount: 0,
		};
	}

	private beginStep(): void {
		if (this.stepping) throw new Error("A simulation step is already pending");
		if (this.faulted) throw new Error("Simulation failed; reset or recreate it before continuing");
		this.stepping = true;
	}

	/** Retain synchronous stepping for non-ML dataset exports and existing agents. */
	runStep(): StepResult {
		if (this.agents.some(agent => agent.asynchronous)) {
			throw new Error("Use runStepAsync for asynchronous agents");
		}
		this.beginStep();
		const step = this.advanceStep();
		let unexpectedPending = false;
		try {
			let next = step.next();
			while (!next.done) {
				if (!Array.isArray(next.value)) {
					// A custom agent may omit its async marker. Keep the lock until its
					// in-flight work settles, even though this caller used the wrong API.
					unexpectedPending = true;
					void Promise.resolve(next.value).then(
						() => { this.stepping = false; },
						() => { this.stepping = false; },
					);
					throw new Error("Use runStepAsync for asynchronous agents");
				}
				next = step.next(next.value);
			}
			return next.value;
		} catch (error) {
			this.faulted = true;
			throw error;
		} finally {
			if (!unexpectedPending) this.stepping = false;
		}
	}

	/** Await one agent at a time; response latency must not change market ordering. */
	async runStepAsync(): Promise<StepResult> {
		this.beginStep();
		const step = this.advanceStep();
		try {
			let next = step.next();
			while (!next.done) next = step.next(await next.value);
			return next.value;
		} catch (error) {
			// Earlier agents may already have traded. Do not retry a partially executed step.
			this.faulted = true;
			throw error;
		} finally {
			this.stepping = false;
		}
	}

	// One generator owns market mutations for both APIs. It yields an agent's decision
	// and resumes with its orders, so sync/async runners cannot drift in trading semantics.
	private *advanceStep(): Generator<NewOrderRequest[] | Promise<NewOrderRequest[]>, StepResult, NewOrderRequest[]> {
		const decisionClock = this.clock + 1;
		const observableContext = { ...this.getObservableContext(), clock: decisionClock };
		// Capture every agent's view before any agent acts or the history advances.
		const observations = new Map<TraderAgent, ObservableSimulatorContext>();
		for (const agent of this.agents) {
			if (agent.observation) {
				const { tradeHistoryLimit, priceHistoryLimit, bookDepth } = agent.observation;
				observations.set(agent, { ...this.getObservableContext(
					tradeHistoryLimit, priceHistoryLimit, bookDepth,
				), clock: decisionClock });
			}
		}
		const reports: ExecutionReport[] = [];
		const stepEvents: SimulationEvent[] = [];

		for (const agent of this.agents) {
			stepEvents.push({
				type: "agent-step",
				timestamp: decisionClock,
				agentId: agent.id,
			});
			const agentContext = this.getAgentContext(
				agent.id, structuredClone(observations.get(agent) ?? observableContext),
			);
			// Public observations stay at the pre-step snapshot; portfolio updates from
			// earlier agents remain visible as in the original sequential simulator.
			agentContext.observationClock = this.clock;

			const orders = yield agent.step(agentContext);
			for (const order of orders) {
				const report = this.exchange.submitOrder(order);
				this.portfolioManager.incrementOrdersSubmitted(agent.id);
				// this.recordOrderSubmission(agent.id, report.orderId);
				reports.push(report);
				stepEvents.push({
					type: "order-submitted",
					timestamp: decisionClock,
					agentId: agent.id,
					orderId: report.orderId,
					report,
				});

				for (const trade of report.trades) {
					this.portfolioManager.applyTrade(trade);
					this.updateStatistics(trade);
				}

				if (report.trades.length > 0) {
					stepEvents.push({
						type: "trade",
						timestamp: decisionClock,
						agentId: agent.id,
						orderId: report.orderId,
						tradeCount: report.trades.length,
					});
				}
			}
		}
		// Publish the completed step only when every agent has finished successfully.
		this.clock = decisionClock;
		this.midPriceHistory.pushBack(observableContext.midPrice);
		for (const event of stepEvents) {
			this.events.pushBack(event);
		}
		this.clampHistorySizes();
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

	updateStatistics(event: TradeEvent): void {
		this.statistics.volume += event.quantity;
		this.statistics.tradeCount += 1;
	}

	clampHistorySizes(): void {
		while (this.midPriceHistory.size() > MAX_MID_PRICE_HISTORY) {
			this.midPriceHistory.popFront();
		}
		while (this.events.size() > MAX_EVENT_HISTORY) {
			this.events.popFront();
		}
	}

	reset(): void {
		if (this.stepping) throw new Error("Cannot reset while a simulation step is pending");
		this.faulted = false;
		this.exchange = new Exchange();
		this.orderParticipants.clear();
		this.clock = 0;
		this.midPriceHistory.clear();
		this.events.clear();
		this.portfolioManager.reset();
		this.initializeParticipantPortfolios();
		this.statistics = {
			volume: 0,
			tradeCount: 0,
		};
	}

	getStatistics(): SimulatorStatistics {
		return { ...this.statistics };
	}

	getEvents(): SimulationEvent[] {
		return this.events.toArray();
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

	getLimitedTradeHistory(depth = 10): LimitedTradeEvent[] {
		return this.getTradeHistory().slice(-depth).map((trade) => ({
			tradeId: trade.tradeId,
			timestamp: trade.timestamp,
			price: trade.price,
			quantity: trade.quantity,
			aggressorSide: trade.aggressorSide,
		}));
	}

	getMidPriceHistory(): number[] {
		return this.midPriceHistory.toArray();
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
		bookDepth = 10,
	): ObservableSimulatorContext {
		if (
			![tradeHistoryLimit, priceHistoryLimit, bookDepth].every(Number.isSafeInteger) ||
			tradeHistoryLimit < 0 || priceHistoryLimit < 0 || bookDepth < 1
		) {
			throw new Error("Observation limits must be non-negative integers and book depth must be positive.");
		}
		const orderBook = this.exchange.getOrderBookSnapshot(bookDepth);

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
			referencePrice: this.referencePrice,
			spread,
			orderBook,
			recentTrades: tradeHistoryLimit === 0
				? [] : this.getLimitedTradeHistory(tradeHistoryLimit),
			recentMidPriceSeries: priceHistoryLimit === 0
				? [] : this.getMidPriceHistory().slice(-priceHistoryLimit),
			orderImbalance: calculateRecentOrderImbalance(orderBook),
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
