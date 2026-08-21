// src/ml/train.ts

import { MLP } from "./models";

export type TradingAction = "SELL" | "HOLD" | "BUY";

export interface TrainingExample {
	features: number[];
	label: TradingAction;
}

export interface TrainingOptions {
	epochs: number;
	learningRate: number;

	shuffle?: boolean;

	onEpochEnd?: (
		epoch: number,
		loss: number,
		accuracy: number,
	) => void;
}

export interface TrainingResult {
	epochs: number;
	finalLoss: number;
	finalAccuracy: number;
	lossHistory: number[];
	accuracyHistory: number[];
}

const ACTIONS: TradingAction[] = [
	"SELL",
	"HOLD",
	"BUY",
];

function actionToIndex(action: TradingAction): number {
	return ACTIONS.indexOf(action);
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

function shuffle<T>(values: T[]): T[] {
	const result = [...values];

	for (let i = result.length - 1; i > 0; i -= 1) {
		const j = Math.floor(Math.random() * (i + 1));

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
): [T[], T[]] {
    const shuffled = shuffle(data);
    const splitIndex = Math.floor(shuffled.length * (1 - testSize));
    return [
        shuffled.slice(0, splitIndex),
        shuffled.slice(splitIndex),
    ];
}

export function train(
	model: MLP,
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
			: shuffle(dataset);

		let totalLoss = 0;
		let correct = 0;

		for (const example of examples) {
			const targetIndex = actionToIndex(example.label);

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
				ACTIONS.length,
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
		finalLoss,
		finalAccuracy,
		lossHistory,
		accuracyHistory,
	};
}

export function test(model: MLP, dataset: TrainingExample[]): {
    loss: number;
    accuracy: number;
} {
    let totalLoss = 0;
    let correct = 0;

    for (const example of dataset) {
        const targetIndex = actionToIndex(example.label);

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

export function train_test(model: MLP, dataset: TrainingExample[], options: TrainingOptions, testSize: number): {
    trainResult: TrainingResult;
    testResult: {
        loss: number;
        accuracy: number;
    };
} {
    const [trainSet, testSet] = train_test_split(dataset, testSize);

    const trainResult = train(model, trainSet, options);
    const testResult = test(model, testSet);

    return {
        trainResult,
        testResult,
    };
}