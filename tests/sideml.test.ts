import { describe, it, expect } from "vitest";
import { MLP, SeededRandom } from "@/simulation";
import { generateSideDataset, train, test, train_test } from '@/simulation';
import type { TrainingExample, TrainingOptions, Model } from '@/simulation';
import { SIDE_ACTIONS as ACTIONS } from '@/simulation';

const seed = 42;
const dataset_size = 100;
const epochs = 300;
const lr = 0.001;

type ModelPreset = {
    dataset: TrainingExample[];
    model: Model;
    options: TrainingOptions;
    testP: number;
};

function createSideModel(inp: number): MLP {
    return new MLP([inp, inp * 4, inp * 2, inp, ACTIONS.length], new SeededRandom(seed));
}

function createDataset(): TrainingExample[] {
    return generateSideDataset({ seed: seed, offset: 42, gap: 3, num: dataset_size });
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

        const result = train(preset.model, ACTIONS, preset.dataset, preset.options);

        // console.log(result.lossHistory);

        expect(result.loss).toBeLessThan(result.lossHistory[0]);
    });
    it ("should achieve reasonable accuracy on a small dataset", () => {

        const preset = createPreset();

        const result = train(preset.model, ACTIONS, preset.dataset, preset.options);

        // console.log(result.accuracyHistory);

        expect(result.accuracy).toBeGreaterThan(0.5);
    });
    it("expect testing to return loss and accuracy", () => {

        const preset = createPreset();

        const result = train(preset.model, ACTIONS, preset.dataset, preset.options);

        // console.log(result.finalAccuracy);

        expect(result.accuracy).toBeGreaterThan(0.5);
    });
    it("expect testing to return loss and accuracy", () => {

        const preset = createPreset();

        const testResult = test(preset.model, ACTIONS, preset.dataset);

        // console.log(trainResult.accuracyHistory);
        // console.log(testResult);

        expect(testResult.loss).toBeGreaterThan(0);
        expect(testResult.accuracy).toBeGreaterThan(0);
        expect(testResult.accuracy).toBeLessThanOrEqual(1);
    });
    it("testing the model should return reasonable results", async () => {

        const preset = createPreset();

        const testResult = train_test(preset.model, ACTIONS, preset.dataset, preset.options, preset.testP).testResult;
        
        // console.log(testResult.accuracy);

        expect(testResult.accuracy).toBeGreaterThan(0.5);
    });
    it("actual training and testing", async () => {

        const preset = createPreset();

        // console.log(preset);

        const result = train_test(preset.model, ACTIONS, preset.dataset, preset.options, preset.testP);

        console.log("Side Model Train Accuracy:", result.trainResult.accuracy);
        console.log("Side Model Test Accuracy:", result.testResult.accuracy);

        await writeToFile(preset.model.toJSON(), "./src/side_model.json");
    });
});