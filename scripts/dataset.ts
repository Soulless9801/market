import type { AgentSide } from '@/simulation';
import { Simulator, buildDefaultAgents} from '@/simulation';
import { buildFeatures } from '@/simulation';

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

import { writeFile } from 'fs/promises';

async function writeToFile(data: string, filePath: string): Promise<void> {
    try {
        await writeFile(filePath, data, 'utf-8');
        // console.log('File written successfully.');
    } catch (error) {
        console.error('Error writing file:', error);
    }
}

const seed = 42;
const offset = 42;
const gap = 3;
const num = 10000;

export async function main() {
    const dataset = generateSideDataset({
        seed: seed,
        offset: offset,
        gap: gap,
        num: num
    });

    console.log(`Generated dataset with ${dataset.length} examples.`);

    const jsonData = JSON.stringify(dataset, null, 2);
    await writeToFile(jsonData, 'datasets/side_dataset.json');
}

main().catch((error) => {
    console.error('Error generating dataset:', error);
    process.exit(1);
});