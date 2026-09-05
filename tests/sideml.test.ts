import { describe, it, expect } from "vitest";
import { MLP, SeededRandom } from "@/simulation";
import { generate, train, test, train_test } from '@/simulation';
import type { TrainingExample, TrainingOptions, TradingAction, Model } from '@/simulation';
    
const ACTIONS: TradingAction[] = ["SELL", "HOLD", "BUY"];

const seed = 42;
const dataset_size = 100;
const epochs = 100;
const lr = 0.005;

type ModelPreset = {
    dataset: TrainingExample[];
    model: Model;
    options: TrainingOptions;
    testP: number;
};

function createSideModel(inp: number): MLP {
    return new MLP([inp, inp * 4, inp * 4, ACTIONS.length], new SeededRandom(seed));
}

function createDataset(): TrainingExample[] {
    return generate({ seed: seed, offset: 42, gap: 3, num: dataset_size });
}

function createTrainingOptions(): TrainingOptions {
    return {
        epochs: epochs,
        learningRate: lr,
        random: new SeededRandom(seed)
    };
}

function createPreset(): ModelPreset {
    const dataset = createDataset();
    const inp = dataset[0].features.length;
    const model = createSideModel(inp);
    const options = createTrainingOptions();
    const testP = 0.2;

    return {
        dataset,
        model,
        options,
        testP,
    };
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

describe("MLP", () => {
    it("should predict output for given input", () => {

        const model = createSideModel(4);

        const output = model.predict([
            0.01,   // short return
            -0.02,  // long return
            0.01,   // spread
            0.35,   // imbalance
        ]);

        expect(output).toHaveLength(ACTIONS.length);
    });
    it("training should reduce loss over epochs", () => {

        const preset = createPreset();
        const dataset = preset.dataset;
        const model = preset.model;
        const options = preset.options;
        // const testP = preset.testP;

        const result = train(model, ACTIONS, dataset, options);

        // console.log(result.lossHistory);

        expect(result.finalLoss).toBeLessThan(result.lossHistory[0]);
    });
    it ("should achieve reasonable accuracy on a small dataset", () => {

        const preset = createPreset();
        const dataset = preset.dataset;
        const model = preset.model;
        const options = preset.options;
        // const testP = preset.testP;

        const result = train(model, ACTIONS, dataset, options);

        // console.log(result.accuracyHistory);

        expect(result.finalAccuracy).toBeGreaterThan(0.5);
    });
    it("expect testing to return loss and accuracy", () => {

        const preset = createPreset();
        const dataset = preset.dataset;
        const model = preset.model;
        const options = preset.options;
        // const testP = preset.testP;

        const result = train(model, ACTIONS, dataset, options);

        // console.log(result.finalAccuracy);

        expect(result.finalAccuracy).toBeGreaterThan(0.5);
    });
    it("expect testing to return loss and accuracy", () => {

        const preset = createPreset();
        const dataset = preset.dataset;
        const model = preset.model;
        // const options = preset.options;
        // const testP = preset.testP;

        const testResult = test(model, ACTIONS, dataset);

        // console.log(trainResult.accuracyHistory);
        // console.log(testResult);

        expect(testResult.loss).toBeGreaterThan(0);
        expect(testResult.accuracy).toBeGreaterThan(0);
        expect(testResult.accuracy).toBeLessThanOrEqual(1);
    });
    it("testing the model should return reasonable results", async () => {

        const preset = createPreset();
        const dataset = preset.dataset;
        const model = preset.model;
        const options = preset.options;
        const testP = preset.testP;

        const testResult = train_test(model, ACTIONS, dataset, options, testP).testResult;
        
        // console.log(testResult.accuracy);

        expect(testResult.accuracy).toBeGreaterThan(0.5);
    });
    it("actual training and testing", async () => {

        const preset = createPreset();
        const dataset = preset.dataset;
        const model = preset.model;
        const options = preset.options;
        const testP = preset.testP;

        // console.log(preset);

        const result = train_test(model, ACTIONS, dataset, options, testP);

        console.log("Side Model Train Accuracy:", result.trainResult.finalAccuracy);
        console.log("Side Model Test Accuracy:", result.testResult.accuracy);

        await writeToFile(model.toJSON(), "./src/side_model.json");
    });
});