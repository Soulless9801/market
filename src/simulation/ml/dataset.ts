import type { TrainingExample } from './train';
import { Simulator } from '../simulator';
import { buildFeatures } from './features';
import { buildDefaultAgents } from '../agents/TraderAgents';

const tradeDepth = 20;
const midPriceDepth = 20;

export function generate(seed: number, offset: number, gap: number, num: number): TrainingExample[] {
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

        let label: "BUY" | "SELL" | "HOLD";
        if (nextMidPrice > currentMidPrice) {
            label = "BUY";
        } else if (nextMidPrice < currentMidPrice) {
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
