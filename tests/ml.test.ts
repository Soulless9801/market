import { describe, it, expect } from "vitest";
import { ConfigManager, CNN, CNNArchitecture, MLP, ModelManager, SeededRandom } from "@/simulation";
import { SIDE_ACTIONS as ACTIONS } from '@/simulation';

const seed = 42;

function createMLPModel(inp: number): MLP {
    const config = ConfigManager.build('mlp', inp, ACTIONS.length);
    return ModelManager.build('mlp', config, new SeededRandom(seed)) as MLP;
}

const cnnConfig = new CNNArchitecture({
    kind: "cnn",
    inputSize: 5,
    outputSize: 2,
    convolutionalLayers: [[2, 3, 1, 1]],
    denseLayers: [3],
});

function createCNNModel(seedValue: number): CNN {
    return new CNN(cnnConfig, new SeededRandom(seedValue));
}

function cnnLoss(candidate: CNN, input: number[], target: number[]): number {
    const probabilities = candidate.predict(input);
    return -target.reduce((sum, value, index) => sum + value * Math.log(probabilities[index]), 0);
}

describe("Models", () => {
    it("MLP should predict output for given input", () => {

        const n = 30;
        const m = 1000;

        // test feature length from 
        for (let i = 1; i <= n; i++) {
            
            const model = createMLPModel(i);

            for (let j = 0; j < m; j++) {
                const feature = Array.from({ length: i }, () => Math.random());
                const output = model.predict(feature);

                expect(output.length).toBe(ACTIONS.length);
            }
        }
    });

    it("CNN produces stable, normalized probabilities and validates input size", () => {
        const probabilities = createCNNModel(1).predict([1, 0, -1, 0.5, 2]);
        expect(probabilities).toHaveLength(2);
        expect(probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
        expect(() => createCNNModel(1).predict([1, 2])).toThrow(/input must contain 5/);
    });

    it("CNN matches a finite-difference gradient for a dense bias", () => {
        const input = [0.2, -0.3, 0.5, 0.1, -0.2];
        const target = [1, 0];
        const epsilon = 1e-5;
        const candidate = createCNNModel(2);
        const serialized = JSON.parse(candidate.toJSON()) as {
            denseLayers: Array<{ biases: number[] }>;
        };
        const parameter = serialized.denseLayers.at(-1)!.biases;
        const plus = createCNNModel(2);
        const minus = createCNNModel(2);
        const plusData = JSON.parse(candidate.toJSON());
        const minusData = JSON.parse(candidate.toJSON());
        plusData.denseLayers.at(-1).biases[0] += epsilon;
        minusData.denseLayers.at(-1).biases[0] -= epsilon;
        plus.fromJSON(JSON.stringify(plusData));
        minus.fromJSON(JSON.stringify(minusData));
        const numerical = (cnnLoss(plus, input, target) - cnnLoss(minus, input, target)) / (2 * epsilon);
        const before = parameter[0];
        candidate.train(input, target, 1e-6);
        const after = JSON.parse(candidate.toJSON()).denseLayers.at(-1).biases[0];
        expect((before - after) / 1e-6).toBeCloseTo(numerical, 4);
    });

    it("CNN trains, serializes, and reproduces deterministically", () => {
        const input = [1, 0, 1, 0, 1];
        const target = [1, 0];
        const first = createCNNModel(7);
        const second = createCNNModel(7);
        const initialLoss = cnnLoss(first, input, target);
        for (let i = 0; i < 25; i++) {
            first.train(input, target, 0.05);
            second.train(input, target, 0.05);
        }
        expect(cnnLoss(first, input, target)).toBeLessThan(initialLoss);
        expect(first.predict(input)).toEqual(second.predict(input));
        const restored = createCNNModel(99);
        restored.fromJSON(first.toJSON());
        expect(restored.predict(input)).toEqual(first.predict(input));
        expect(createCNNModel(8).predict(input)).not.toEqual(createCNNModel(7).predict(input));
    });

    it("registers CNN architectures with ModelManager", () => {
        const architecture = ConfigManager.build("cnn", 5, 2);
        const built = ModelManager.build("cnn", architecture, new SeededRandom(3));
        expect(built.predict([0, 1, 0, 1, 0])).toHaveLength(2);
    });
});