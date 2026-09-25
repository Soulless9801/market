import { SeededRandom } from "@/simulation/agents";

// interface for model
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

// architecture interface/metadata for models
export interface ModelArchitecture {

	// model kind (e.g., "mlp", "cnn")
	kind: string;

	// validate architecture parameters
	validate(): void;

	// check equality with another architecture
	equals(other: ModelArchitecture): boolean;
}

// interface for architecture generation
export interface ConfigGenerator {
	// generate architecture from inp
	g(inp: number, out: number): ModelConfig;
}

export interface ModelConfig {
	kind: string;
}

// model configuration interface for mlp
export interface MLPConfig extends ModelConfig {
	layers: number[];
}

// mlp architecture implementation
export class MLPArchitecture implements ModelArchitecture {

	public readonly kind = "mlp" as const;
	public readonly layers: number[];

	constructor(config: MLPConfig) {
		const layers = config.layers;
		this.layers = [...layers];
		this.validate();
	}

	//@override
	validate(): void {
		if (this.layers.length < 2) throw new Error("MLP architecture requires at least an input and output layer.");
		if (this.layers.some((size) => !Number.isInteger(size) || size <= 0)) throw new Error("MLP architecture layer sizes must be positive integers.");
	}

	//@override
	equals(other: ModelArchitecture): boolean {
		if (other.kind !== this.kind) return false;
		const otherMLP = other as MLPArchitecture;
		if (this.layers.length !== otherMLP.layers.length) return false;
		for (let i = 0; i < this.layers.length; i++) {
			if (this.layers[i] !== otherMLP.layers[i]) return false;
		}
		return true;
	}
}

// class for generating MLP architecture
export class MLPGenerator implements ConfigGenerator {

	//@override
	g(inp: number, out: number): MLPConfig {
		return {
			kind: "mlp",
			layers: [inp, Math.max(32, inp * 4), Math.max(16, inp * 2), 16, out],
		};
	}
}

// dense layer class for MLP
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

	private readonly architecture: MLPArchitecture;
	private readonly layers: DenseLayer[];

	constructor(
		config: ModelConfig,
        random: SeededRandom
	) {

		// deep copy of architecture
		this.architecture = new MLPArchitecture(config as MLPConfig);

		const layers = this.architecture.layers;

		this.layers = [];

		for (
			let i = 0;
			i < layers.length - 1;
			i++
		) {
			this.layers.push(
				new DenseLayer(
					layers[i],
					layers[i + 1],
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

		const architecture = this.architecture;

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

		const parsed = data as {
			architecture: MLPArchitecture;
			weights: number[][][];
			biases: number[][];
		};

		if (
			!parsed.architecture ||
			!parsed.weights ||
			!parsed.biases
		) {
			throw new Error(
				"Invalid model JSON.",
			);
		}

		if (!this.architecture.equals(parsed.architecture)) {
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
					parsed.weights[i].length ||
				layer.weights[0].length !==
					parsed.weights[i][0].length
			) {
				throw new Error(
					"Model weights do not match.",
				);
			}

			layer.weights = parsed.weights[i];
			layer.biases = parsed.biases[i];
		}
	}
}


// interface for cnn config
export interface CNNConfig extends ModelConfig {
	inputSize: number;
	outputSize: number;
	convolutionalLayers: number[][];
	denseLayers: number[];
	poolingSize?: number;
}

// interface for cnn architecture
export class CNNArchitecture implements ModelArchitecture {

	public readonly kind = "cnn" as const;
	public readonly inputSize: number;
	public readonly outputSize: number;
	public readonly convolutionalLayers: number[][];
	public readonly denseLayers: number[];
	public readonly poolingSize?: number;

	constructor(config: CNNConfig) {
		this.inputSize = config.inputSize;
		this.outputSize = config.outputSize;
		this.convolutionalLayers = config.convolutionalLayers.map(
			(layer) => [...layer],
		);
		this.denseLayers = [...config.denseLayers];
		this.poolingSize = config.poolingSize ?? 1;
		this.validate();
	}

	//@override
	validate(): void {
		if (
			!Number.isInteger(this.inputSize) ||
			this.inputSize <= 0 ||
			!Number.isInteger(this.outputSize) ||
			this.outputSize <= 0 ||
			this.convolutionalLayers.length === 0 ||
			this.denseLayers.some(
				(size) => !Number.isInteger(size) || size <= 0,
			)
		) {
			throw new Error(
				"Invalid CNN configuration.",
			);
		}
		for (const layer of this.convolutionalLayers) {
			if (layer.length !== 4) {
				throw new Error(
					"Invalid CNN convolutional layer configuration.",
				);
			}
			for (const param of layer) {
				if (!Number.isInteger(param) || param <= 0) {
					throw new Error(
						"Invalid CNN convolutional layer parameters.",
					);
				}
			}
		}
		if (
			this.poolingSize !== undefined &&
			(!Number.isInteger(this.poolingSize) || this.poolingSize <= 0)
		) {
			throw new Error(
				"Invalid CNN pooling size.",
			);
		}
	}

	//@override
	equals(other: ModelArchitecture): boolean {
		if (other.kind !== this.kind) {
			return false;
		}

		const otherCNN = other as CNNArchitecture;
		if (
			this.inputSize !== otherCNN.inputSize ||
			this.outputSize !== otherCNN.outputSize ||
			this.poolingSize !== otherCNN.poolingSize ||
			this.convolutionalLayers.length !==
				otherCNN.convolutionalLayers.length ||
			this.denseLayers.length !== otherCNN.denseLayers.length
		) {
			return false;
		}
		for (
			let layerIndex = 0;
			layerIndex < this.convolutionalLayers.length;
			layerIndex++
		) {
			for (
				let parameterIndex = 0;
				parameterIndex < this.convolutionalLayers[layerIndex].length;
				parameterIndex++
			) {
				if (
					this.convolutionalLayers[layerIndex][parameterIndex] !==
					otherCNN.convolutionalLayers[layerIndex][parameterIndex]
				) {
					return false;
				}
			}
		}
		for (
			let layerIndex = 0;
			layerIndex < this.denseLayers.length;
			layerIndex++
		) {
			if (this.denseLayers[layerIndex] !== otherCNN.denseLayers[layerIndex]) {
				return false;
			}
		}
		return true;
	}
}

export class CNNGenerator implements ConfigGenerator {
	g(
		inp: number,
		out: number,
	): CNNConfig {
		return {
			kind: "cnn",
			inputSize: inp,
			outputSize: out,
			convolutionalLayers: [[4, 3, 1, 1], [8, 3, 1, 1]],
			denseLayers: [inp > 2 ? 8 : 4],
			poolingSize: 1,
		};
	}
}

export class CNNConvolutionLayer {
	public readonly filters: number;
	public readonly kernelSize: number;
	public readonly stride: number;
	public readonly padding: number;
	public weights: number[][];
	public biases: number[];

	constructor(
		filters: number,
		kernelSize: number,
		stride = 1,
		padding = 0,
	) {
		this.filters = filters;
		this.kernelSize = kernelSize;
		this.stride = stride;
		this.padding = padding;
		this.weights = [];
		this.biases = [];
	}

	initialize(inputChannels: number, random: SeededRandom): void {
		const fanIn = inputChannels * this.kernelSize;
		this.weights = Array.from(
			{ length: this.filters },
			() =>
				Array.from(
					{ length: fanIn },
					() => randomNormal(random) * Math.sqrt(2 / fanIn),
				),
		);
		this.biases = new Array(this.filters).fill(0);
	}
}

export class CNNPoolCache {
	public readonly input: number[];
	public readonly output: number[];
	public readonly indices: number[];

	constructor(input: number[], output: number[], indices: number[]) {
		this.input = input;
		this.output = output;
		this.indices = indices;
	}
}

function assertFiniteValues(values: number[], name: string): void {
	if (values.some((value) => !Number.isFinite(value))) {
		throw new Error(`${name} contains a non-finite value.`);
	}
}

function randomNormal(random: SeededRandom): number {
	let u = 0;
	let v = 0;
	while (u === 0) {
		u = random.next();
	}
	while (v === 0) {
		v = random.next();
	}
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export class CNN implements Model {

	private readonly architecture: CNNArchitecture;
	private readonly convolutionalLayers: CNNConvolutionLayer[];
	private readonly denseLayers: DenseLayer[];
	private readonly poolingSize: number;

	constructor(config: ModelConfig, random: SeededRandom) {
		// deep copy
		this.architecture = new CNNArchitecture(config as CNNConfig);
		this.architecture.validate();

		this.poolingSize = this.architecture.poolingSize ?? 1;
		this.convolutionalLayers = [];

		for (
			let layerIndex = 0;
			layerIndex < this.architecture.convolutionalLayers.length;
			layerIndex++
		) {
			const layerConfig = this.architecture.convolutionalLayers[layerIndex];
			const inputChannels = layerIndex === 0
				? 1
				: this.architecture.convolutionalLayers[layerIndex - 1][0];
			const layer = new CNNConvolutionLayer(
				layerConfig[0],
				layerConfig[1],
				layerConfig[2],
				layerConfig[3],
			);
			layer.initialize(inputChannels, random);
			this.convolutionalLayers.push(layer);
		}
		const sizes = this.activationSizes();
		let denseInput =
			sizes[sizes.length - 1].channels *
			sizes[sizes.length - 1].length;
		this.denseLayers = [];
		for (
			const size of [
				...this.architecture.denseLayers,
				this.architecture.outputSize,
			]
		) {
			this.denseLayers.push(
				new DenseLayer(denseInput, size, random),
			);
			denseInput = size;
		}
	}

	private activationSizes(): Array<{ channels: number; length: number }> {
		let length = this.architecture.inputSize;
		const sizes: Array<{ channels: number; length: number }> = [];
		for (const layer of this.convolutionalLayers) {
			length =
				Math.floor(
					(length + 2 * layer.padding - layer.kernelSize) /
						layer.stride,
				) + 1;
			if (length <= 0) {
				throw new Error(
					"CNN convolution produces an invalid output size.",
				);
			}
			const channels = layer.filters;
			if (this.poolingSize > 1) {
				length = Math.floor(length / this.poolingSize);
				if (length <= 0) {
					throw new Error(
						"CNN pooling produces an invalid output size.",
					);
				}
			}
			sizes.push({ channels, length });
		}
		return sizes;
	}

	private convolution(
		input: number[],
		inputChannels: number,
		layer: CNNConvolutionLayer,
	): number[] {
		const inputLength = input.length / inputChannels;
		const outputLength =
			Math.floor(
				(inputLength + 2 * layer.padding - layer.kernelSize) /
					layer.stride,
			) + 1;
		const output = new Array(layer.filters * outputLength).fill(0);
		for (
			let filter = 0;
			filter < layer.filters;
			filter++
		) {
			for (
				let position = 0;
				position < outputLength;
				position++
			) {
				let value = layer.biases[filter];
				for (
					let channel = 0;
					channel < inputChannels;
					channel++
				) {
					for (
						let kernel = 0;
						kernel < layer.kernelSize;
						kernel++
					) {
						const source = position * layer.stride + kernel - layer.padding;
						if (source >= 0 && source < inputLength) {
							value +=
								layer.weights[filter][channel * layer.kernelSize + kernel] *
								input[channel * inputLength + source];
						}
					}
				}
				output[filter * outputLength + position] = Math.max(0, value);
			}
		}
		return output;
	}

	private pool(
		input: number[],
		channels: number,
		length: number,
	): CNNPoolCache {
		if (this.poolingSize === 1) {
			return new CNNPoolCache(
				input,
				[...input],
				input.map((_, index) => index),
			);
		}
		const outputLength = Math.floor(length / this.poolingSize);
		const output = new Array(channels * outputLength);
		const indices = new Array(output.length);
		for (
			let channel = 0;
			channel < channels;
			channel++
		) {
			for (
				let position = 0;
				position < outputLength;
				position++
			) {
				let best = channel * length + position * this.poolingSize;
				for (
					let offset = 1;
					offset < this.poolingSize;
					offset++
				) {
					const candidate = best + offset;
					if (input[candidate] > input[best]) {
						best = candidate;
					}
				}
				output[channel * outputLength + position] = input[best];
				indices[channel * outputLength + position] = best;
			}
		}
		return new CNNPoolCache(input, output, indices);
	}

	private forward(input: number[]): {
		output: number[];
		convolutions: number[][];
		pools: CNNPoolCache[];
		denseInputs: number[][];
	} {
		let activation = input;
		let channels = 1;
		const convolutions: number[][] = [];
		const pools: CNNPoolCache[] = [];
		for (const layer of this.convolutionalLayers) {
			const convolved = this.convolution(activation, channels, layer);
			convolutions.push(convolved);
			const pooled = this.pool(convolved, layer.filters, convolved.length / layer.filters);
			pools.push(pooled);
			activation = pooled.output;
			channels = layer.filters;
		}
		const denseInputs = [activation];
		for (let i = 0; i < this.denseLayers.length - 1; i++) {
			activation = applyRelu(this.denseLayers[i].forward(activation));
			denseInputs.push(activation);
		}
		return {
			output: this.denseLayers[this.denseLayers.length - 1].forward(activation),
			convolutions,
			pools,
			denseInputs,
		};
	}

	//@override
	predict(input: number[]): number[] {
		this.validateInput(input);
		const logits = this.forward(input).output;
		return softmax(logits);
	}

	//@override
	train(input: number[], target: number[], learningRate: number): void {
		this.validateInput(input);
		if (target.length !== this.architecture.outputSize) {
			throw new Error(
				"CNN target size does not match output size.",
			);
		}
		assertFiniteValues(target, "CNN target");
		if (!Number.isFinite(learningRate) || learningRate <= 0) {
			throw new Error(
				"CNN learning rate must be positive and finite.",
			);
		}
		const cache = this.forward(input);
		const probabilities = softmax(cache.output);
		let gradients = probabilities.map(
			(probability, index) => probability - target[index],
		);
		for (
			let layerIndex = this.denseLayers.length - 1;
			layerIndex >= 0;
			layerIndex--
		) {
			const layer = this.denseLayers[layerIndex];
			const layerInput = cache.denseInputs[layerIndex];
			const previous = new Array(layerInput.length).fill(0);
			for (
				let neuron = 0;
				neuron < layer.weights.length;
				neuron++
			) {
				for (
					let index = 0;
					index < layerInput.length;
					index++
				) {
					previous[index] +=
						gradients[neuron] * layer.weights[neuron][index];
				}
				for (
					let index = 0;
					index < layerInput.length;
					index++
				) {
					layer.weights[neuron][index] -=
						learningRate *
						gradients[neuron] *
						layerInput[index];
				}
				layer.biases[neuron] -= learningRate * gradients[neuron];
			}
			if (layerIndex > 0) {
				const preActivation = this.denseLayers[
					layerIndex - 1
				].forward(cache.denseInputs[layerIndex - 1]);
				gradients = previous.map(
					(value, index) =>
						value * (preActivation[index] > 0 ? 1 : 0),
				);
			} else {
				gradients = previous;
			}
		}
		let activation = input;
		const activations: number[][] = [input];
		let channels = 1;
		for (const layer of this.convolutionalLayers) {
			const convolved = this.convolution(activation, channels, layer);
			const pooled = this.pool(convolved, layer.filters, convolved.length / layer.filters);
			activation = pooled.output;
			channels = layer.filters;
			activations.push(activation);
		}
		for (
			let layerIndex = this.convolutionalLayers.length - 1;
			layerIndex >= 0;
			layerIndex--
		) {
			const layer = this.convolutionalLayers[layerIndex];
			const oldWeights = layer.weights.map((row) => [...row]);
			const inputActivation = activations[layerIndex];
			const inputChannels = layerIndex === 0
				? 1
				: this.convolutionalLayers[layerIndex - 1].filters;
			const inputLength = inputActivation.length / inputChannels;
			const convolved = cache.convolutions[layerIndex];
			const convolvedLength = convolved.length / layer.filters;
			const pool = cache.pools[layerIndex];
			const next = new Array(inputActivation.length).fill(0);
			const gradConvolved = new Array(convolved.length).fill(0);
			for (
				let index = 0;
				index < gradients.length;
				index++
			) {
				gradConvolved[pool.indices[index]] += gradients[index];
			}
			for (
				let filter = 0;
				filter < layer.filters;
				filter++
			) {
				for (
					let position = 0;
					position < convolvedLength;
					position++
				) {
					const reluGradient =
						convolved[filter * convolvedLength + position] > 0
							? gradConvolved[filter * convolvedLength + position]
							: 0;
					layer.biases[filter] -= learningRate * reluGradient;
					for (
						let channel = 0;
						channel < inputChannels;
						channel++
					) {
						for (
							let kernel = 0;
							kernel < layer.kernelSize;
							kernel++
						) {
							const source =
								position * layer.stride! +
								kernel -
								layer.padding!;
							if (source >= 0 && source < inputLength) {
								const weightIndex = channel * layer.kernelSize + kernel;
								layer.weights[filter][weightIndex] -=
									learningRate *
									reluGradient *
									inputActivation[channel * inputLength + source];
								next[channel * inputLength + source] +=
									reluGradient * oldWeights[filter][weightIndex];
							}
						}
					}
				}
			}
			gradients = next;
		}
	}

	private validateInput(input: number[]): void {
		if (input.length !== this.architecture.inputSize) {
			throw new Error(
				`CNN input must contain ${this.architecture.inputSize} values.`,
			);
		}
		assertFiniteValues(input, "CNN input");
	}

	//@override
	toJSON(): string {

		const architecture = this.architecture;

		const convolutionalLayers = this.convolutionalLayers.map((layer) => ({
			filters: layer.filters,
			kernelSize: layer.kernelSize,
			stride: layer.stride,
			padding: layer.padding,
			weights: layer.weights,
			biases: layer.biases,
		}));

		const denseLayers = this.denseLayers.map((layer) => ({
			weights: layer.weights,
			biases: layer.biases,
		}));

		return JSON.stringify({
			architecture,
			convolutionalLayers,
			denseLayers,
		}, null, 2);
	}

	//@override
	fromJSON(json: string): void {

		const data = JSON.parse(json);

		const parsed = data as {
			architecture: CNNArchitecture;
			convolutionalLayers: CNNConvolutionLayer[];
			denseLayers: DenseLayer[];
		};

		if (!this.architecture.equals(parsed.architecture)) {
			throw new Error(
				"CNN model configuration does not match.",
			);
		}

		for (
			let layerIndex = 0;
			layerIndex < this.convolutionalLayers.length;
			layerIndex++
		) {
			const source = parsed.convolutionalLayers[layerIndex];
			const target = this.convolutionalLayers[layerIndex];
			if (
				!Array.isArray(source.weights) ||
				source.weights.length !== target.weights.length ||
				source.weights.some(
					(row, index) =>
						!Array.isArray(row) ||
						row.length !== target.weights[index].length,
				) ||
				!Array.isArray(source.biases) ||
				source.biases.length !== target.biases.length
			) {
				throw new Error(
					"CNN convolution parameters do not match.",
				);
			}
			target.weights = source.weights.map((row) => [...row]);
			target.biases = [...source.biases];
			assertFiniteValues(target.weights.flat(), "CNN convolution weights");
			assertFiniteValues(target.biases, "CNN convolution biases");
		}

		for (
			let layerIndex = 0;
			layerIndex < this.denseLayers.length;
			layerIndex++
		) {
			const source = parsed.denseLayers[layerIndex];
			const target = this.denseLayers[layerIndex];
			if (
				!source ||
				!Array.isArray(source.weights) ||
				source.weights.length !== target.weights.length ||
				source.weights.some(
					(row, index) =>
						!Array.isArray(row) ||
						row.length !== target.weights[index].length,
				) ||
				!Array.isArray(source.biases) ||
				source.biases.length !== target.biases.length
			) {
				throw new Error(
					"CNN dense parameters do not match.",
				);
			}
			target.weights = source.weights.map((row) => [...row]);
			target.biases = [...source.biases];
			assertFiniteValues(target.weights.flat(), "CNN dense weights");
			assertFiniteValues(target.biases, "CNN dense biases");
		}
	}
}

// type for model constructors
type ModelConstructor = new (config: ModelConfig, random: SeededRandom) => Model;

// model manager class
export class ModelManager {

	// registry for model constructors
	private static readonly registry = new Map<string, ModelConstructor>([
		["mlp", MLP],
		["cnn", CNN],
	]);

	// build a model from the registry
	static build(
		modelName: string,
		config: ModelConfig,
		random: SeededRandom,
	): Model {
		const builder = this.registry.get(modelName);
		if (!builder) {
			throw new Error(
				`No model builder registered for model: ${modelName}`,
			);
		}
		if (config.kind !== modelName) {
			throw new Error(
				`Architecture kind ${config.kind} does not match model: ${modelName}`,
			);
		}
		return new builder(config, random);
	}
}

// constructor type for configs
type ConfigConstructor = new () => ConfigGenerator;

// config manager class
export class ConfigManager {

	// registry for config constructors
	private static readonly registry = new Map<string, ConfigConstructor>([
		["mlp", MLPGenerator],
		["cnn", CNNGenerator],
	]);

	// build config from the registry
	static build(
		model: string,
		inp: number,
		out: number,
	): ModelConfig {
		const builder = this.registry.get(model);
		if (!builder) {
			throw new Error(
				`No architecture builder registered for model: ${model}`,
			);
		}
		return new builder().g(inp, out);
	}
}
