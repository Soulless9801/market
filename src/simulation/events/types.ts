import type { ExecutionReport } from "../../engine";

export type SimulationEventType = "agent-step" | "order-submitted" | "trade";

export interface SimulationEvent {
	type: SimulationEventType;
	timestamp: number;
	agentId: string;
	orderId?: string;
	report?: ExecutionReport;
	tradeCount?: number;
}
