import { Exchange, type ExecutionReport, type OrderBookSnapshot, type TradeEvent } from "../../engine";
import type { SimulationEvent } from "../events";
import type { SimulationContext, TraderAgent } from "../agents";

export interface SimulatorOptions {
	exchange?: Exchange;
	agents: TraderAgent[];
	referencePrice?: number;
}

export interface ParticipantStats {
	agentId: string;
	ordersSubmitted: number;
	inventory: number;
}

export interface StepResult {
	step: number;
	reports: ExecutionReport[];
	events: SimulationEvent[];
	trades: TradeEvent[];
}

export class Simulator {
	private exchange: Exchange;
	private readonly agents: TraderAgent[];
	private readonly referencePrice: number;
	private readonly events: SimulationEvent[] = [];
	private readonly participantStats = new Map<string, ParticipantStats>();
	private readonly orderParticipants = new Map<string, string>();
	private clock = 0;

	constructor(options: SimulatorOptions) {
		this.exchange = options.exchange ?? new Exchange();
		this.agents = options.agents;
		this.referencePrice = options.referencePrice ?? 100;
		this.initializeParticipantStats();
	}

	runStep(): StepResult {
		this.clock += 1;
		const context = this.createContext();
		const reports: ExecutionReport[] = [];
		const stepEvents: SimulationEvent[] = [];

		for (const agent of this.agents) {
			stepEvents.push({
				type: "agent-step",
				timestamp: this.clock,
				agentId: agent.id,
			});

			const orders = agent.step(context);
			for (const order of orders) {
				const report = this.exchange.submitOrder(order);
				this.recordOrderSubmission(agent.id, report.orderId);
				reports.push(report);
				stepEvents.push({
					type: "order-submitted",
					timestamp: this.clock,
					agentId: agent.id,
					orderId: report.orderId,
					report,
				});

				for (const trade of report.trades) {
					this.applyTrade(trade);
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
		this.participantStats.clear();
		this.orderParticipants.clear();
		this.clock = 0;
		this.initializeParticipantStats();
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

	getParticipantStats(): ParticipantStats[] {
		return Array.from(this.participantStats.values());
	}

	getClock(): number {
		return this.clock;
	}

	private initializeParticipantStats(): void {
		for (const agent of this.agents) {
			this.participantStats.set(agent.id, {
				agentId: agent.id,
				ordersSubmitted: 0,
				inventory: 0,
			});
		}
	}

	private recordOrderSubmission(agentId: string, orderId: string): void {
		this.orderParticipants.set(orderId, agentId);
		const participant = this.participantStats.get(agentId);
		if (participant) {
			participant.ordersSubmitted += 1;
		}
	}

	private applyTrade(trade: TradeEvent): void {
		const makerParticipantId = this.orderParticipants.get(trade.makerOrderId);
		const takerParticipantId = this.orderParticipants.get(trade.takerOrderId);
		if (makerParticipantId) {
			this.updateInventory(makerParticipantId, trade.buyOrderId === trade.makerOrderId ? trade.quantity : -trade.quantity);
		}
		if (takerParticipantId) {
			this.updateInventory(takerParticipantId, trade.buyOrderId === trade.takerOrderId ? trade.quantity : -trade.quantity);
		}
	}

	private updateInventory(agentId: string, delta: number): void {
		const participant = this.participantStats.get(agentId);
		if (participant) {
			participant.inventory += delta;
		}
	}

	private createContext(): SimulationContext {
		const snapshot = this.exchange.getOrderBookSnapshot();
		const trades = this.exchange.getTradeHistory();
		const lastTrade = trades[trades.length - 1];
		const midPrice = this.calculateMidPrice(snapshot);
		return {
			clock: this.clock,
			snapshot,
			midPrice,
			lastTradePrice: lastTrade?.price,
			tradeCount: trades.length,
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
}
