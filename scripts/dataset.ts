import type { AgentSide } from '@/simulation';
import { FeatureManager, NormalizerManager, Simulator, buildDefaultAgents} from '@/simulation';

const tradeDepth = 20;
const midPriceDepth = 20;

export interface TrainingExample {
	features: number[];
	label: AgentSide;
}

type DatasetOptions = {
    model: string;
    seed: number;
    offset: number;
    gap: number;
    num: number;
};

const threshold = 1e-2;

export function generateSideDataset(options: DatasetOptions): TrainingExample[] {
    const { seed, offset, gap, num } = options;
    const examples: TrainingExample[] = [];
    const builder = FeatureManager.create(options.model);
    const trajectoryCount = 20;
    const examplesPerTrajectory = Math.ceil(num / trajectoryCount);

    for (let trajectoryIndex = 0; trajectoryIndex < trajectoryCount && examples.length < num; trajectoryIndex++) {
        const simulator = new Simulator({
            agents: buildDefaultAgents(seed + trajectoryIndex * 9973),
        });
        while (simulator.getClock() < offset) {
            simulator.runStep();
        }

        for (let exampleIndex = 0; exampleIndex < examplesPerTrajectory && examples.length < num; exampleIndex++) {
            const context = simulator.getObservableContext(tradeDepth, midPriceDepth);
            const features = builder.build(context);

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
const gap = 10;
const num = 10000;

export async function main() {

    const args = process.argv.slice(2);

	const modelStr = args[0];

    const dataset = generateSideDataset({
        model: modelStr,
        seed: seed,
        offset: offset,
        gap: gap,
        num: num
    });

    console.log(`Generated dataset with ${dataset.length} examples.`);

    const normalizer = NormalizerManager.getNormalizer("gaussian", dataset.map(example => example.features));

    const normalizerData = normalizer.toJSON();
    await writeToFile(normalizerData, `datasets/side_normalizer_${modelStr}.json`);

    for (const example of dataset) {
        example.features = normalizer.transform(example.features);
    }

    const jsonData = JSON.stringify(dataset, null, 2);
    await writeToFile(jsonData, `datasets/side_dataset_${modelStr}.json`);
}

main().catch((error) => {
    console.error('Error generating dataset:', error);
    process.exit(1);
});