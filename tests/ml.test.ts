import { describe, it, expect } from "vitest";
import { MLP, SeededRandom } from "@/simulation";
import { generate, train, test, train_test } from '@/simulation';
import type { TrainingExample, TrainingOptions } from '@/simulation';
    
function createSideModel(seed: number, inp: number): MLP {
    const random = new SeededRandom(seed);
    return new MLP([inp, 16, 16, 3], random);
}

function generateDataset(): TrainingExample[] {
    return generate({ seed: 42, offset: 42, gap: 3, num: 100 });
}

function generateTrainingOptions(seed: number): TrainingOptions {
    return {
        epochs: 100,
        learningRate: 0.005,
        random: new SeededRandom(seed),
    };
}

import { writeFile } from 'fs/promises';

async function writeToFile(data: string, filePath: string): Promise<void> {
    try {
        await writeFile(filePath, data, 'utf-8');
        console.log('File written successfully.');
    } catch (error) {
        console.error('Error writing file:', error);
    }
}

describe("MLP", () => {
    it("should predict output for given input", () => {
        const model = createSideModel(42, 4);

        const output = model.predict([
            0.01,   // short return
            -0.02,  // long return
            0.01,   // spread
            0.35,   // imbalance
        ]);

        expect(output).toHaveLength(3);
    });
    it("training should reduce loss over epochs", () => {
        const dataset: TrainingExample[] = generateDataset();

        const inp = dataset[0].features.length;

        const model = createSideModel(42, inp);

        const options = generateTrainingOptions(42);

        const result = train(model, dataset, options);

        // console.log(result.lossHistory);

        expect(result.finalLoss).toBeLessThan(result.lossHistory[0]);
    });
    it ("should achieve reasonable accuracy on a small dataset", () => {
        const dataset: TrainingExample[] = generateDataset();

        const inp = dataset[0].features.length;

        const model = createSideModel(42, inp);

        const options = generateTrainingOptions(42);

        const result = train(model, dataset, options);

        // console.log(result.accuracyHistory);

        expect(result.finalAccuracy).toBeGreaterThan(0.5);
    });
    it("expect testing to return loss and accuracy", () => {
        const dataset: TrainingExample[] = generateDataset();

        const inp = dataset[0].features.length;

        const model = createSideModel(42, inp);

        const options = generateTrainingOptions(42);
        const result = train(model, dataset, options);

        // console.log(result.finalAccuracy);

        expect(result.finalAccuracy).toBeGreaterThan(0.5);
    });
    it("expect testing to return loss and accuracy", () => {

        
       const dataset: TrainingExample[] = generateDataset();

        const inp = dataset[0].features.length;

        const model = createSideModel(42, inp);

        const testResult = test(model, dataset);

        // console.log(trainResult.accuracyHistory);
        // console.log(testResult);

        expect(testResult.loss).toBeGreaterThan(0);
        expect(testResult.accuracy).toBeGreaterThan(0);
        expect(testResult.accuracy).toBeLessThanOrEqual(1);
    });
    it("testing the model should return reasonable results", async () => {

        const dataset: TrainingExample[] = generateDataset();

        const inp = dataset[0].features.length;

        const model = createSideModel(42, inp);

        const options = generateTrainingOptions(42);

        const testP = 0.2;

        const testResult = train_test(model, dataset, options, testP).testResult;
        
        // console.log(testResult.accuracy);

        // console.log(model);

        expect(testResult.accuracy).toBeGreaterThan(0.5);
    });
    it("actual training and testing should return reasonable results", async () => {

        const dataset: TrainingExample[] = generateDataset();

        const inp = dataset[0].features.length;

        const model = createSideModel(42, inp);

        const options = generateTrainingOptions(42);

        const testP = 0.2;

        const result = train_test(model, dataset, options, testP);
        
        // console.log(result.testResult.accuracy);

        expect(result.testResult.accuracy).toBeGreaterThan(0.5);

        await writeToFile(model.toJSON(), "./src/model.json");
    });
});