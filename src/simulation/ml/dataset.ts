import type { TrainingExample } from './train';
import { Simulator } from '../simulator';
import { buildFeatures, writeToFile } from './features';
import { buildDefaultAgents } from '../agents/TraderAgents';

const tradeDepth = 20;
const midPriceDepth = 20;

export type DatasetOptions = {
    seed: number;
    offset: number;
    gap: number;
    num: number;
};

export function generate(options: DatasetOptions): TrainingExample[] {
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

export function download(options: DatasetOptions, filePath: string): void {
    const examples = generate(options);
    const dataString = JSON.stringify(examples, null, 2);
    writeToFile(dataString, filePath);
}
