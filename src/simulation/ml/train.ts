// src/ml/train.ts

import type { Model } from "./models";
import type { TrainingExample } from "./dataset";
import type { AgentSide } from "@/simulation";
import { SeededRandom } from "@/simulation";

export interface TrainingOptions {
	epochs: number;
	learningRate: number;

	shuffle?: boolean;

	onEpochEnd?: (
		epoch: number,
		loss: number,
		accuracy: number,
	) => void;

	random: SeededRandom;
}

export interface TrainingResult {
	epochs: number;
	loss: number;
	accuracy: number;
	lossHistory: number[];
	accuracyHistory: number[];
}

export interface TestResult {
	loss: number;
	accuracy: number;
}

function softmax(logits: number[]): number[] {
	const maxLogit = Math.max(...logits);

	const exponentials = logits.map((logit) =>
		Math.exp(logit - maxLogit),
	);

	const sum = exponentials.reduce(
		(total, value) => total + value,
		0,
	);

	return exponentials.map((value) => value / sum);
}

function crossEntropyLoss(
	probabilities: number[],
	targetIndex: number,
): number {
	const probability = Math.max(
		probabilities[targetIndex],
		1e-12,
	);

	return -Math.log(probability); // bits of information lmao
}

function argmax(values: number[]): number {
	let bestIndex = 0;

	for (let i = 1; i < values.length; i += 1) {
		if (values[i] > values[bestIndex]) {
			bestIndex = i;
		}
	}

	return bestIndex;
}

function shuffle<T>(values: T[], random: SeededRandom): T[] {

	const result = [...values];

	for (let i = result.length - 1; i > 0; i -= 1) {

		const j = Math.floor(random.next() * (i + 1));

		[result[i], result[j]] = [
			result[j],
			result[i],
		];
	}

	return result;
}

export function train_test_split<T>(
    data: T[],
    testSize: number,
    random: SeededRandom,
): [T[], T[]] {
    const shuffled = shuffle(data, random);
	// console.log(shuffled[0]);
    const splitIndex = Math.floor(shuffled.length * (1 - testSize));
    return [
        shuffled.slice(0, splitIndex),
        shuffled.slice(splitIndex),
    ];
}

export function train(
	model: Model,
	actions: AgentSide[],
	dataset: TrainingExample[],
	options: TrainingOptions,
): TrainingResult {
	if (dataset.length === 0) {
		throw new Error("Cannot train on an empty dataset.");
	}

	if (options.epochs <= 0) {
		throw new Error("epochs must be greater than zero.");
	}

	if (options.learningRate <= 0) {
		throw new Error("learningRate must be greater than zero.");
	}

	const lossHistory: number[] = [];
	const accuracyHistory: number[] = [];

	let finalLoss = 0;
	let finalAccuracy = 0;

	for (
		let epoch = 0;
		epoch < options.epochs;
		epoch += 1
	) {
		const examples = options.shuffle === false
			? dataset
			: shuffle(dataset, options.random);

		let totalLoss = 0;
		let correct = 0;

		for (const example of examples) {
			const targetIndex = actions.indexOf(example.label);

			if (targetIndex === -1) {
				throw new Error(
					`Unknown trading action: ${example.label}`,
				);
			}

			const logits = model.predict(
				example.features,
			);

			const probabilities = softmax(logits);

			totalLoss += crossEntropyLoss(
				probabilities,
				targetIndex,
			);

			const prediction = argmax(probabilities);

			if (prediction === targetIndex) {
				correct += 1;
			}

			const target = new Array(
				actions.length,
			).fill(0);

			target[targetIndex] = 1;

			model.train(
				example.features,
				target,
				options.learningRate,
			);
		}

		finalLoss = totalLoss / examples.length;
		finalAccuracy = correct / examples.length;

		lossHistory.push(finalLoss);
		accuracyHistory.push(finalAccuracy);

		options.onEpochEnd?.(
			epoch + 1,
			finalLoss,
			finalAccuracy,
		);
	}

	return {
		epochs: options.epochs,
		loss: finalLoss,
		accuracy: finalAccuracy,
		lossHistory,
		accuracyHistory,
	};
}

export function test(model: Model, actions: AgentSide[], dataset: TrainingExample[]): {
    loss: number;
    accuracy: number;
} {
    let totalLoss = 0;
    let correct = 0;

    for (const example of dataset) {
        const targetIndex = actions.indexOf(example.label);

        if (targetIndex === -1) {
            throw new Error(
                `Unknown trading action: ${example.label}`,
            );
        }

        const logits = model.predict(
            example.features,
        );

        const probabilities = softmax(logits);

        totalLoss += crossEntropyLoss(
            probabilities,
            targetIndex,
        );

        const prediction = argmax(probabilities);

        if (prediction === targetIndex) {
            correct += 1;
        }
    }

    return {
        loss: totalLoss / dataset.length,
        accuracy: correct / dataset.length,
    };
}

export function train_test(model: Model, actions: AgentSide[], dataset: TrainingExample[], options: TrainingOptions, testSize: number): {
    trainResult: TrainingResult;
    testResult: TestResult;
} {

    const [trainSet, testSet] = train_test_split(dataset, testSize, options.random);
	
    const trainResult = train(model, actions, trainSet, options);
    const testResult = test(model, actions, testSet);

    return {
        trainResult,
        testResult,
    };
}
