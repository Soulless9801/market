import { SeededRandom } from "@/simulation/agents";

export interface Model {

	// predict probabilities for each class
	predict(input: number[]): number[];

	// backpropagation training step
	train(
		input: number[],
		target: number[],
		learningRate: number,
	): void;

	// convert model to JSON
	toJSON(): string;

	// load model from JSON
	fromJSON(json: string): void;
}

export class DenseLayer {

	weights: number[][];
	biases: number[];

	constructor(
		inputSize: number,
		outputSize: number,
		random: SeededRandom,
	) {
		const std = Math.sqrt(2 / inputSize);

		this.weights = Array.from(
			{ length: outputSize },
			() =>
				Array.from(
					{ length: inputSize },
					() => this.randomNormal(random) * std,
				),
		);

		this.biases = new Array(outputSize).fill(0);
	}

	forward(input: number[]): number[] {
		return this.weights.map((row, i) => {
			let value = this.biases[i];

			for (let j = 0; j < input.length; j++) {
				value += row[j] * input[j];
			}

			return value;
		});
	}

	private randomNormal(random: SeededRandom): number {
		// Box-Muller transform
		let u = 0;
		let v = 0;

		while (u === 0) {
			u = random.next();
		}

		while (v === 0) {
			v = random.next();
		}

		return Math.sqrt(-2 * Math.log(u)) *
			Math.cos(2 * Math.PI * v);
	}
}

function applyRelu(values: number[]): number[] {
	return values.map((value) =>
		Math.max(0, value),
	);
}

function reluDerivative(
	values: number[],
): number[] {
	return values.map((value) =>
		value > 0 ? 1 : 0,
	);
}

function softmax(values: number[]): number[] {
	const max = Math.max(...values);

	const exponentials = values.map((value) =>
		Math.exp(value - max),
	);

	const sum = exponentials.reduce(
		(total, value) => total + value,
		0,
	);

	return exponentials.map(
		(value) => value / sum,
	);
}

export class MLP implements Model {

	private readonly layers: DenseLayer[];

	constructor(
		architecture: number[],
        random: SeededRandom
	) {
		if (architecture.length < 2) throw new Error("MLP requires at least an input and output layer.");

		this.layers = [];

		for (
			let i = 0;
			i < architecture.length - 1;
			i++
		) {
			this.layers.push(
				new DenseLayer(
					architecture[i],
					architecture[i + 1],
					random,
				),
			);
		}
	}

	// @override
	predict(input: number[]): number[] {

		let activation = input;

		for (
			let layerIndex = 0;
			layerIndex < this.layers.length;
			layerIndex++
		) {
			const layer = this.layers[layerIndex];

			activation = layer.forward(
				activation,
			);

			// ReLU on every layer except output.
			if (
				layerIndex <
				this.layers.length - 1
			) {
				activation =
					applyRelu(activation);
			}
		}

		return activation;
	}

	// @override
	train(
		input: number[],
		target: number[],
		learningRate: number,
	): void {


		const preActivations: number[][] = [];
		const activations: number[][] = [];

		let activation = input;

		activations.push(input);

		for (
			let layerIndex = 0;
			layerIndex < this.layers.length;
			layerIndex++
		) {
			const layer = this.layers[layerIndex];

			const preActivation =
				layer.forward(activation);

			preActivations.push(
				preActivation,
			);

			const isOutputLayer =
				layerIndex ===
				this.layers.length - 1;

			activation = isOutputLayer
				? preActivation
				: applyRelu(
						preActivation,
					);

			activations.push(activation);
		}

		const output = activation;

		const probabilities =
			softmax(output);

		let gradients =
			probabilities.map(
				(probability, i) =>
					probability - target[i],
			);

		for (
			let layerIndex =
				this.layers.length - 1;
			layerIndex >= 0;
			layerIndex--
		) {
			const layer =
				this.layers[layerIndex];

			const layerInput =
				activations[layerIndex];

			for (
				let neuronIndex = 0;
				neuronIndex <
				layer.weights.length;
				neuronIndex++
			) {
				for (
					let inputIndex = 0;
					inputIndex <
					layer.weights[
						neuronIndex
					].length;
					inputIndex++
				) {
					layer.weights[
						neuronIndex
					][inputIndex] -=
						learningRate *
						gradients[
							neuronIndex
						] *
						layerInput[
							inputIndex
						];
				}

				layer.biases[
					neuronIndex
				] -=
					learningRate *
					gradients[
						neuronIndex
					];
			}

			if (layerIndex === 0) {
				break;
			}

			const previousLayerSize =
				this.layers[
					layerIndex - 1
				].weights.length;

			const previousGradients =
				new Array(
					previousLayerSize,
				).fill(0);

			for (
				let previousNeuron = 0;
				previousNeuron <
				previousLayerSize;
				previousNeuron++
			) {
				for (
					let neuronIndex = 0;
					neuronIndex <
					layer.weights.length;
					neuronIndex++
				) {
					previousGradients[
						previousNeuron
					] +=
						gradients[
							neuronIndex
						] *
						layer.weights[
							neuronIndex
						][previousNeuron];
				}
			}

			const previousPreActivation =
				preActivations[
					layerIndex - 1
				];

			const derivative =
				reluDerivative(
					previousPreActivation,
				);

			gradients =
				previousGradients.map(
					(gradient, i) =>
						gradient *
						derivative[i],
				);
		}
	}

	//@override
	toJSON(): string {
		const architecture = this.layers.map(
			(layer) => layer.weights[0].length,
		);

		architecture.push(
			this.layers[
				this.layers.length - 1
			].weights.length,
		);

		const weights = this.layers.map(
			(layer) => layer.weights,
		);

		const biases = this.layers.map(
			(layer) => layer.biases,
		);

		return JSON.stringify({
			architecture,
			weights,
			biases,
		}, null, 2);
	}

	//@override
	fromJSON(json: string): void {
		const data = JSON.parse(json);

		if (
			!data.architecture ||
			!data.weights ||
			!data.biases
		) {
			throw new Error(
				"Invalid model JSON.",
			);
		}

		const architecture: number[] =
			data.architecture;

		if (
			architecture.length !==
			this.layers.length + 1
		) {
			throw new Error(
				"Model architecture does not match.",
			);
		}

		for (
			let i = 0;
			i < this.layers.length;
			i++
		) {
			const layer = this.layers[i];

			if (
				layer.weights.length !==
					data.weights[i].length ||
				layer.weights[0].length !==
					data.weights[i][0].length
			) {
				throw new Error(
					"Model weights do not match.",
				);
			}

			layer.weights = data.weights[i];
			layer.biases = data.biases[i];
		}
	}
}

// export class CNN implements Model {
// 	predict(input: number[]): number[] {
// 		throw new Error("Method not implemented.");
// 	}
// 	train(input: number[], target: number[], learningRate: number): void {
// 		throw new Error("Method not implemented.");
// 	}
// 	toJSON(): string {
// 		throw new Error("Method not implemented.");
// 	}
// 	fromJSON(json: string): void {
// 		throw new Error("Method not implemented.");
// 	}
// }

// constructor type for models
type ModelConstructor = new (architecture: number[], random: SeededRandom) => Model;

// model manager class
export class ModelManager {

	// registry for model constructors
	private static readonly registry = new Map<string, ModelConstructor>([
		['mlp', MLP],
		// ['cnn', CNN]
	]); 

	// build a model from the registry
	static build(modelName: string, architecture: number[], random: SeededRandom): Model {
		const builder = this.registry.get(modelName);
		if (!builder) throw new Error(`No model builder registered for model: ${modelName}`);
		return new builder(architecture, random);
	}
}

// interface for architecture generation
export interface ArchitectureGenerator {
	// generate architecture from inp
	g(inp: number, out: number): number[];
}

// class for generating MLP architecture
export class MLPGenerator implements ArchitectureGenerator {

	g(inp: number, out: number): number[] {
		return [inp, inp * 4, inp * 2, inp, out];
	}
}

// constructor type for architectures
type ArchitectureConstructor = new () => ArchitectureGenerator;

// architecture manager class
export class ArchitectureManager {
	
	// registry for architecture constructors
	private static readonly registry = new Map<string, ArchitectureConstructor>([
		['mlp', MLPGenerator],
	]);

	// build architecture from the registry
	static build(model: string, inp: number, out: number): number[] {
		const builder = this.registry.get(model);
		if (!builder) throw new Error(`No architecture builder registered for model: ${model}`);
		return new builder().g(inp, out);
	}
}
