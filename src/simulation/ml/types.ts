import type { AgentSide } from "@/simulation/agents";

export interface TrainingExample {
	features: number[];
	label: AgentSide;
}

export const SIDE_ACTIONS: AgentSide[] = ["BUY", "SELL", "HOLD"];
