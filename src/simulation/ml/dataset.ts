import type { AgentSide } from '@/simulation';
import { Simulator, buildDefaultAgents} from '@/simulation';
import { buildFeatures } from './features';

const tradeDepth = 20;
const midPriceDepth = 20;

export interface TrainingExample {
	features: number[];
	label: AgentSide;
}

export type DatasetOptions = {
    seed: number;
    offset: number;
    gap: number;
    num: number;
};

const threshold = 1e-2;

export function generateSideDataset(options: DatasetOptions): TrainingExample[] {
    const { seed, offset, gap, num } = options;
    const examples: TrainingExample[] = [];
    const simulator = new Simulator({
        agents: buildDefaultAgents(seed),
    });
    while (simulator.getClock() < offset) {
        simulator.runStep();
    }
    for (let i = 0; i < num; i++) {
        const context = simulator.getObservableContext(tradeDepth, midPriceDepth);
        const features = buildFeatures(context);

        const currentMidPrice = context.midPrice;
        for (let j = 0; j < gap; j++) {
            simulator.runStep();
        }
        const nextContext = simulator.getObservableContext(0, 0); // only need midprice
        const nextMidPrice = nextContext.midPrice;

        let label: AgentSide;
        if (nextMidPrice - currentMidPrice > threshold) {
            label = "BUY";
        } else if (currentMidPrice - nextMidPrice > threshold) {
            label = "SELL";
        } else {
            label = "HOLD";
        }

        examples.push({
            features,
            label,
        });
    }

    return examples;
}
