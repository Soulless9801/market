export interface Optimizer {
	readonly name: string;

	beginStep(): void;

	update(
		parameterId: string,
		value: number,
		gradient: number,
		learningRate: number,
		isWeight?: boolean,
	): number;

	reset(): void;
}

function validateLearningRate(learningRate: number): void {
	if (!Number.isFinite(learningRate) || learningRate <= 0) {
		throw new Error("Learning rate must be positive and finite.");
	}
}

export class SGDOptimizer implements Optimizer {
	readonly name = "sgd";
	beginStep(): void {}

	update(_parameterId: string, value: number, gradient: number, learningRate: number): number {
		validateLearningRate(learningRate);
		return value - learningRate * gradient;
	}

	reset(): void {}
}

export class MomentumOptimizer implements Optimizer {
	readonly name = "momentum";
	private readonly velocity = new Map<string, number>();
	private readonly momentum: number;

	constructor(momentum = 0.9) {
		if (!Number.isFinite(momentum) || momentum < 0 || momentum >= 1) {
			throw new Error("Momentum must be in the range [0, 1).");
		}
		this.momentum = momentum;
	}

	beginStep(): void {}

	update(parameterId: string, value: number, gradient: number, learningRate: number): number {
		validateLearningRate(learningRate);
		const velocity = this.momentum * (this.velocity.get(parameterId) ?? 0) - learningRate * gradient;
		this.velocity.set(parameterId, velocity);
		return value + velocity;
	}

	reset(): void {
		this.velocity.clear();
	}
}

export class AdagradOptimizer implements Optimizer {
	readonly name = "adagrad";
	private readonly squaredGradients = new Map<string, number>();
	private readonly epsilon: number;

	constructor(epsilon = 1e-8) {
		this.epsilon = epsilon;
	}

	beginStep(): void {}

	update(parameterId: string, value: number, gradient: number, learningRate: number): number {
		validateLearningRate(learningRate);
		const squaredGradient = (this.squaredGradients.get(parameterId) ?? 0) + gradient * gradient;
		this.squaredGradients.set(parameterId, squaredGradient);
		return value - learningRate * gradient / (Math.sqrt(squaredGradient) + this.epsilon);
	}

	reset(): void {
		this.squaredGradients.clear();
	}
}

export class RMSPropOptimizer implements Optimizer {
	readonly name = "rmsprop";
	private readonly squaredGradients = new Map<string, number>();
	private readonly decay: number;
	private readonly epsilon: number;

	constructor(decay = 0.99, epsilon = 1e-8) {
		if (!Number.isFinite(decay) || decay < 0 || decay >= 1) {
			throw new Error("RMSProp decay must be in the range [0, 1).");
		}
		this.decay = decay;
		this.epsilon = epsilon;
	}

	beginStep(): void {}

	update(parameterId: string, value: number, gradient: number, learningRate: number): number {
		validateLearningRate(learningRate);
		const previous = this.squaredGradients.get(parameterId) ?? 0;
		const squaredGradient = this.decay * previous + (1 - this.decay) * gradient * gradient;
		this.squaredGradients.set(parameterId, squaredGradient);
		return value - learningRate * gradient / (Math.sqrt(squaredGradient) + this.epsilon);
	}

	reset(): void {
		this.squaredGradients.clear();
	}
}

export class AdamOptimizer implements Optimizer {
	readonly name: string = "adam";
	protected readonly firstMoments = new Map<string, number>();
	protected readonly secondMoments = new Map<string, number>();
	protected step = 0;
	protected readonly beta1: number;
	protected readonly beta2: number;
	protected readonly epsilon: number;

	constructor(beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
		if (beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1) {
			throw new Error("Adam beta values must be in the range [0, 1).");
		}
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = epsilon;
	}

	beginStep(): void {
		this.step += 1;
	}

	update(parameterId: string, value: number, gradient: number, learningRate: number): number {
		validateLearningRate(learningRate);
		const firstMoment = this.beta1 * (this.firstMoments.get(parameterId) ?? 0) + (1 - this.beta1) * gradient;
		const secondMoment = this.beta2 * (this.secondMoments.get(parameterId) ?? 0) + (1 - this.beta2) * gradient * gradient;
		this.firstMoments.set(parameterId, firstMoment);
		this.secondMoments.set(parameterId, secondMoment);
		const correctedFirst = firstMoment / (1 - this.beta1 ** this.step);
		const correctedSecond = secondMoment / (1 - this.beta2 ** this.step);
		return value - learningRate * correctedFirst / (Math.sqrt(correctedSecond) + this.epsilon);
	}

	reset(): void {
		this.firstMoments.clear();
		this.secondMoments.clear();
		this.step = 0;
	}
}

export class AdamWOptimizer extends AdamOptimizer {
	readonly name = "adamw";
	private readonly weightDecay: number;

	constructor(weightDecay = 1e-2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
		super(beta1, beta2, epsilon);
		if (!Number.isFinite(weightDecay) || weightDecay < 0) {
			throw new Error("AdamW weight decay must be non-negative.");
		}
		this.weightDecay = weightDecay;
	}

	update(parameterId: string, value: number, gradient: number, learningRate: number, isWeight = true): number {
		const updated = super.update(parameterId, value, gradient, learningRate);
		return isWeight ? updated - learningRate * this.weightDecay * value : updated;
	}
}
