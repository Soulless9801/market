import { describe, it, expect } from "vitest";
import { MLP, SeededRandom } from "../simulation";
import { generate, train, test, train_test } from '../simulation';
import type { TrainingExample, TrainingOptions } from '../simulation';

function createModel(seed: number): MLP {
    const random = new SeededRandom(seed);
    return new MLP([4, 8, 3], random);
}

describe("MLP", () => {
    it("should predict output for given input", () => {
        const model = createModel(42);

        const output = model.predict([
            0.01,   // short return
            -0.02,  // long return
            0.01,   // spread
            0.35,   // imbalance
        ]);

        expect(output).toHaveLength(3);
    });
    it("training should reduce loss over epochs", () => {
        const model = createModel(42);

        const dataset: TrainingExample[] = generate({ seed: 42, offset: 42, gap: 3, num: 10 });

        const options: TrainingOptions = {
            epochs: 10,
            learningRate: 0.01,
            random: new SeededRandom(42),
        };

        const result = train(model, dataset, options);

        // console.log(result.lossHistory);

        expect(result.finalLoss).toBeLessThan(result.lossHistory[0]);
    });
    it ("should achieve reasonable accuracy on a small dataset", () => {
        const model = createModel(42);

        const dataset: TrainingExample[] = generate({ seed: 42, offset: 42, gap: 3, num: 100 });

        const options: TrainingOptions = {
            epochs: 20,
            learningRate: 0.01,
            random: new SeededRandom(42),
        };

        const result = train(model, dataset, options);

        // console.log(result.accuracyHistory);

        expect(result.finalAccuracy).toBeGreaterThan(0.5);
    });
    it("expect testing to return loss and accuracy", () => {
        const model = createModel(42);

        const dataset: TrainingExample[] = generate({ seed: 42, offset: 42, gap: 3, num: 100 });

        const testResult= test(model, dataset);

        // console.log(trainResult.accuracyHistory);
        // console.log(testResult);

        expect(testResult.loss).toBeGreaterThan(0);
        expect(testResult.accuracy).toBeGreaterThan(0);
        expect(testResult.accuracy).toBeLessThanOrEqual(1);
    });
    it("testing the model should return reasonable results", () => {
        const model = createModel(42);

        const dataset: TrainingExample[] = generate({ seed: 42, offset: 42, gap: 3, num: 100 });

        const options: TrainingOptions = {
            epochs: 100,
            learningRate: 0.1,
            random: new SeededRandom(42),
        };

        const testP = 0.2;

        const testResult = train_test(model, dataset, options, testP).testResult;
        
        // console.log(testResult);

        // console.log(model);

        expect(testResult.accuracy).toBeGreaterThan(0.5);
    });
});