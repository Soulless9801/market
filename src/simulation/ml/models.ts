import { SeededRandom } from "@/simulation/agents";

export interface Model {
	predict(input: number[]): number[];

	train(
		input: number[],
		target: number[],
		learningRate: number,
	): void;

	toJSON(): string;

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
		if (architecture.length < 2) {
			throw new Error(
				"MLP requires at least an input and output layer.",
			);
		}

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

	predictProbabilities(
		input: number[],
	): number[] {
		return softmax(
			this.predict(input),
		);
	}

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