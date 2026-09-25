export interface LearningRateScheduler {
	readonly name: string;

	getLearningRate(
		epoch: number,
		totalEpochs: number,
		baseLearningRate: number,
	): number;
}

function validateBaseLearningRate(learningRate: number): void {
	if (!Number.isFinite(learningRate) || learningRate <= 0) {
		throw new Error("Base learning rate must be positive and finite.");
	}
}

function clampEpoch(epoch: number, totalEpochs: number): number {
	if (!Number.isFinite(epoch) || epoch < 0) return 0;
	if (!Number.isFinite(totalEpochs) || totalEpochs <= 0) return 0;
	return Math.min(epoch, totalEpochs);
}

export class ConstantLearningRate implements LearningRateScheduler {
	readonly name = "constant";

	getLearningRate(_epoch: number, _totalEpochs: number, baseLearningRate: number): number {
		validateBaseLearningRate(baseLearningRate);
		return baseLearningRate;
	}
}

export class StepDecayScheduler implements LearningRateScheduler {
	readonly name = "step-decay";
	private readonly dropEvery: number;
	private readonly decayFactor: number;

	constructor(dropEvery: number, decayFactor = 0.5) {
		if (!Number.isInteger(dropEvery) || dropEvery <= 0) throw new Error("dropEvery must be a positive integer.");
		if (!Number.isFinite(decayFactor) || decayFactor <= 0 || decayFactor > 1) throw new Error("decayFactor must be in the range (0, 1].");
		this.dropEvery = dropEvery;
		this.decayFactor = decayFactor;
	}

	getLearningRate(epoch: number, _totalEpochs: number, baseLearningRate: number): number {
		validateBaseLearningRate(baseLearningRate);
		return baseLearningRate * this.decayFactor ** Math.floor(Math.max(0, epoch) / this.dropEvery);
	}
}

export class ExponentialDecayScheduler implements LearningRateScheduler {
	readonly name = "exponential-decay";
	private readonly decayRate: number;

	constructor(decayRate: number) {
		if (!Number.isFinite(decayRate) || decayRate < 0) throw new Error("decayRate must be non-negative.");
		this.decayRate = decayRate;
	}

	getLearningRate(epoch: number, _totalEpochs: number, baseLearningRate: number): number {
		validateBaseLearningRate(baseLearningRate);
		return baseLearningRate * Math.exp(-this.decayRate * Math.max(0, epoch));
	}
}

export class LinearDecayScheduler implements LearningRateScheduler {
	readonly name = "linear-decay";
	private readonly minimumLearningRate: number;

	constructor(minimumLearningRate = 0) {
		if (!Number.isFinite(minimumLearningRate) || minimumLearningRate < 0) throw new Error("minimumLearningRate must be non-negative.");
		this.minimumLearningRate = minimumLearningRate;
	}

	getLearningRate(epoch: number, totalEpochs: number, baseLearningRate: number): number {
		validateBaseLearningRate(baseLearningRate);
		const progress = clampEpoch(epoch, totalEpochs) / Math.max(1, totalEpochs);
		return this.minimumLearningRate + (baseLearningRate - this.minimumLearningRate) * (1 - progress);
	}
}

export class CosineAnnealingScheduler implements LearningRateScheduler {
	readonly name = "cosine-annealing";
	private readonly minimumLearningRate: number;

	constructor(minimumLearningRate = 0) {
		if (!Number.isFinite(minimumLearningRate) || minimumLearningRate < 0) throw new Error("minimumLearningRate must be non-negative.");
		this.minimumLearningRate = minimumLearningRate;
	}

	getLearningRate(epoch: number, totalEpochs: number, baseLearningRate: number): number {
		validateBaseLearningRate(baseLearningRate);
		const progress = clampEpoch(epoch, totalEpochs) / Math.max(1, totalEpochs);
		return this.minimumLearningRate + 0.5 * (baseLearningRate - this.minimumLearningRate) * (1 + Math.cos(Math.PI * progress));
	}
}

export class PolynomialDecayScheduler implements LearningRateScheduler {
	readonly name = "polynomial-decay";
	private readonly power: number;
	private readonly minimumLearningRate: number;

	constructor(power = 1, minimumLearningRate = 0) {
		if (!Number.isFinite(power) || power < 0) throw new Error("power must be non-negative.");
		if (!Number.isFinite(minimumLearningRate) || minimumLearningRate < 0) throw new Error("minimumLearningRate must be non-negative.");
		this.power = power;
		this.minimumLearningRate = minimumLearningRate;
	}

	getLearningRate(epoch: number, totalEpochs: number, baseLearningRate: number): number {
		validateBaseLearningRate(baseLearningRate);
		const progress = clampEpoch(epoch, totalEpochs) / Math.max(1, totalEpochs);
		return this.minimumLearningRate + (baseLearningRate - this.minimumLearningRate) * (1 - progress) ** this.power;
	}
}

export class WarmupCosineScheduler implements LearningRateScheduler {
	readonly name = "warmup-cosine";
	private readonly warmupEpochs: number;
	private readonly minimumLearningRate: number;
	private readonly warmupStartLearningRate: number;

	constructor(warmupEpochs: number, minimumLearningRate = 0, warmupStartLearningRate = 0) {
		if (!Number.isInteger(warmupEpochs) || warmupEpochs < 0) throw new Error("warmupEpochs must be a non-negative integer.");
		this.warmupEpochs = warmupEpochs;
		this.minimumLearningRate = minimumLearningRate;
		this.warmupStartLearningRate = warmupStartLearningRate;
	}

	getLearningRate(epoch: number, totalEpochs: number, baseLearningRate: number): number {
		validateBaseLearningRate(baseLearningRate);
		if (epoch < this.warmupEpochs && this.warmupEpochs > 0) {
			return this.warmupStartLearningRate + (baseLearningRate - this.warmupStartLearningRate) * epoch / this.warmupEpochs;
		}
		const decayEpoch = Math.max(0, epoch - this.warmupEpochs);
		const decayEpochs = Math.max(1, totalEpochs - this.warmupEpochs);
		const progress = Math.min(1, decayEpoch / decayEpochs);
		return this.minimumLearningRate + 0.5 * (baseLearningRate - this.minimumLearningRate) * (1 + Math.cos(Math.PI * progress));
	}
}
