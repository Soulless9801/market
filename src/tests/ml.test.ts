import { describe, it, expect } from "vitest";
import { MLP } from "../simulation";
import { generate, train, test, train_test } from '../simulation';
import type { TrainingExample, TrainingOptions } from '../simulation';

describe("MLP", () => {
    it("should predict output for given input", () => {
        const model = new MLP();

        const output = model.predict([
            0.01,   // short return
            -0.02,  // long return
            0.01,   // spread
            0.35,   // imbalance
        ]);

        expect(output).toHaveLength(3);
    });
    it("training should reduce loss over epochs", () => {
        const model = new MLP();

        const dataset: TrainingExample[] = generate(42, 42, 3, 10);

        const options: TrainingOptions = {
            epochs: 10,
            learningRate: 0.01,
        };

        const result = train(model, dataset, options);

        // console.log(result.lossHistory);

        expect(result.finalLoss).toBeLessThan(result.lossHistory[0]);
    });
    it ("should achieve reasonable accuracy on a small dataset", () => {
        const model = new MLP();

        const dataset: TrainingExample[] = generate(42, 42, 3, 100);

        const options: TrainingOptions = {
            epochs: 20,
            learningRate: 0.01,
        };

        const result = train(model, dataset, options);

        // console.log(result.accuracyHistory);

        expect(result.finalAccuracy).toBeGreaterThanOrEqual(0.6);
    });
    it("expect testing to return loss and accuracy", () => {
        const model = new MLP();

        const dataset: TrainingExample[] = generate(42, 42, 3, 100);

        const testResult= test(model, dataset);

        // console.log(trainResult.accuracyHistory);
        // console.log(testResult);

        expect(testResult.loss).toBeGreaterThan(0);
        expect(testResult.accuracy).toBeGreaterThan(0);
        expect(testResult.accuracy).toBeLessThanOrEqual(1);
    });
    it("testing the model should return reasonable results", () => {
        const model = new MLP();

        const dataset: TrainingExample[] = generate(42, 42, 3, 100);

        const options: TrainingOptions = {
            epochs: 100,
            learningRate: 0.01,
        };

        const testP = 0.2;

        const { trainResult, testResult } = train_test(model, dataset, options, testP);
        
        // console.log(trainResult.accuracyHistory);
        // console.log(testResult);

        // console.log(model);

        // expect(trainResult.finalAccuracy).toBeGreaterThanOrEqual(0.6);
        expect(testResult.accuracy).toBeGreaterThanOrEqual(0.6);
    });
});