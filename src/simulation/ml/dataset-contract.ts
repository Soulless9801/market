import type { AgentSide } from "../agents";
import { SIDE_ACTIONS } from "./types";

export type DatasetModel = "mlp" | "cnn";

export interface DatasetOptions {
	model: DatasetModel;
	seed: number;
	warmupSteps: number;
	horizonSteps: number;
	sampleCount: number;
	trajectoryCount: number;
}

export const DEFAULT_DATASET_OPTIONS = {
	seed: 42,
	warmupSteps: 42,
	horizonSteps: 10,
	sampleCount: 10000,
	trajectoryCount: 20,
};

export const HISTORY_STEPS = 20;
export const TRADE_HISTORY_LIMIT = 20;
export const TRAJECTORY_SEED_STRIDE = 9973;
export const SIDE_TARGET_THRESHOLD = 0.01;

/** Only public features and offline provenance/target cross the export boundary. */
export interface DatasetExample {
	simulationSeed: number;
	trajectoryIndex: number;
	step: number;
	input: number[];
	/** Class index in metadata.classNames; never a model input. */
	target: number;
}

function assertInteger(
	value: number,
	name: string,
	minimum: number,
	maximum = Number.MAX_SAFE_INTEGER,
): void {
	if (
		!Number.isSafeInteger(value) ||
		value < minimum ||
		value > maximum
	) {
		throw new Error(
			`${name} must be an integer in [${minimum}, ${maximum}].`,
		);
	}
}

export function validateDatasetOptions(options: DatasetOptions): void {
	if (options.model !== "mlp" && options.model !== "cnn")
		throw new Error("model must be mlp or cnn.");
	assertInteger(options.seed, "seed", 0, 0xffffffff);
	assertInteger(
		options.warmupSteps,
		"warmupSteps",
		options.model === "mlp" ? HISTORY_STEPS : 0,
	);
	assertInteger(options.horizonSteps, "horizonSteps", 1);
	assertInteger(options.sampleCount, "sampleCount", 1);
	assertInteger(
		options.trajectoryCount,
		"trajectoryCount",
		1,
		0x100000000,
	);
	const samplesPerTrajectory = Math.ceil(
		options.sampleCount / options.trajectoryCount,
	);
	if (
		!Number.isSafeInteger(
			options.warmupSteps +
				samplesPerTrajectory * options.horizonSteps,
		)
	) {
		throw new Error(
			"The final label step exceeds the safe integer range.",
		);
	}
}

/** Mirrors SeededRandom's uint32 arithmetic, without unsafe intermediate products. */
export function trajectorySeed(
	baseSeed: number,
	trajectoryIndex: number,
): number {
	return (
		(baseSeed +
			(trajectoryIndex % 0x100000000) *
				TRAJECTORY_SEED_STRIDE) >>>
		0
	);
}

export function classifyMidPriceChange(
	current: number,
	future: number,
): AgentSide {
	if (!Number.isFinite(current) || !Number.isFinite(future))
		throw new Error("Label prices must be finite.");
	const change = future - current;
	if (!Number.isFinite(change))
		throw new Error("Label price change must be finite.");
	return change > SIDE_TARGET_THRESHOLD
		? "BUY"
		: change < -SIDE_TARGET_THRESHOLD
			? "SELL"
			: "HOLD";
}

function featureNames(model: DatasetModel): string[] {
	if (model === "cnn")
		return Array.from(
			{ length: 30 },
			(_, i) => `midprice_window_${i}`,
		);
	return [
		"log_midprice_over_reference",
		"log_history_max_over_reference",
		"log_history_min_over_reference",
		"log_midprice_over_last",
		"log_midprice_over_middle_observation",
		"log_midprice_over_first",
		"spread_over_midprice",
		"book_bid_volume_fraction",
		"recent_trade_count_imbalance",
		"log1p_rms_log_return",
		"top_level_quantity_imbalance",
		"log_best_bid_over_midprice",
		"log_best_ask_over_midprice",
		"log1p_best_bid_quantity",
		"log1p_best_ask_quantity",
		...Array.from(
			{ length: HISTORY_STEPS },
			(_, i) => `chronological_log_return_${i}`,
		),
		...Array.from(
			{ length: 5 },
			(_, i) => `depth_quantity_imbalance_${i}`,
		),
	];
}

/** Construct canonical, deterministic metadata; no timestamps or machine paths. */
export function createDatasetContract(options: DatasetOptions) {
	validateDatasetOptions(options);
	const samplesPerTrajectory = Math.ceil(
		options.sampleCount / options.trajectoryCount,
	);
	return {
		schemaVersion: 1 as const,
		datasetType: "market-side-classification" as const,
		inputShape: [options.model === "mlp" ? 40 : 30],
		targetType: "classification" as const,
		targetEncoding: "class-index" as const,
		numClasses: SIDE_ACTIONS.length,
		classNames: [...SIDE_ACTIONS],
		features: {
			version: 1,
			kind: options.model,
			names: featureNames(options.model),
			normalization: "none" as const,
			historyOrder: "oldest-to-newest" as const,
			priceHistoryLimit: HISTORY_STEPS,
			tradeHistoryLimit: TRADE_HISTORY_LIMIT,
			bookDepth: 10,
			padding:
				options.model === "mlp"
					? "none-require-20-history-observations"
					: "prepend-current-midprice-to-length-30",
			description:
				options.model === "mlp"
					? "15 market summaries, 20 chronological log returns (last ends at current midprice), 5 signed depth imbalances; middle price is chronological, volatility is RMS return; invalid log-price ratios and non-finite summaries become zero."
					: "30 price levels: up to 20 historical pre-order midprices, left-padded with the current midprice; padding is not additional historical data.",
		},
		label: {
			quantity: "future-midprice-minus-current-midprice",
			threshold: SIDE_TARGET_THRESHOLD,
			thresholdUnit: "absolute-price",
			horizonSteps: options.horizonSteps,
			buy: "change > threshold",
			sell: "change < -threshold",
			hold: "-threshold <= change <= threshold",
		},
		generation: {
			version: 1,
			model: options.model,
			seed: options.seed,
			warmupSteps: options.warmupSteps,
			horizonSteps: options.horizonSteps,
			sampleCount: options.sampleCount,
			trajectoryCount: options.trajectoryCount,
			samplesPerTrajectory,
			actualTrajectoryCount: Math.ceil(
				options.sampleCount / samplesPerTrajectory,
			),
			trajectorySeedStride: TRAJECTORY_SEED_STRIDE,
			seedArithmetic: "uint32",
			agentPopulation: "buildDefaultAgents-v1-no-ml-agent",
			referencePrice: 100,
			observationTiming:
				"after-completed-step-before-label-horizon",
			historyTiming:
				"one-pre-order-midprice-per-completed-step",
			rowOrder: "trajectory-index-then-observation-step",
		},
	};
}

export type DatasetContract = ReturnType<typeof createDatasetContract>;
export interface DatasetMetadata extends DatasetContract {
	datasetFile: "dataset.jsonl";
	sampleCount: number;
	classCounts: Record<AgentSide, number>;
	/** SHA-256 of the exact UTF-8 dataset.jsonl bytes, including line endings. */
	sha256: string;
}

export function validateDatasetExample(
	value: unknown,
	contract: DatasetContract,
): asserts value is DatasetExample {
	if (!value || typeof value !== "object" || Array.isArray(value))
		throw new Error("Example must be an object.");
	const keys = [
		"simulationSeed",
		"trajectoryIndex",
		"step",
		"input",
		"target",
	];
	if (
		Object.keys(value).length !== keys.length ||
		!keys.every((key) => Object.hasOwn(value, key))
	) {
		throw new Error(
			"Example must contain exactly simulationSeed, trajectoryIndex, step, input and target.",
		);
	}
	const row = value as DatasetExample;
	const generation = contract.generation;
	assertInteger(
		row.trajectoryIndex,
		"trajectoryIndex",
		0,
		generation.actualTrajectoryCount - 1,
	);
	assertInteger(row.simulationSeed, "simulationSeed", 0, 0xffffffff);
	if (
		row.simulationSeed !==
		trajectorySeed(generation.seed, row.trajectoryIndex)
	)
		throw new Error("Incorrect trajectory seed.");
	assertInteger(row.step, "step", generation.warmupSteps);
	const sampleIndex =
		(row.step - generation.warmupSteps) / generation.horizonSteps;
	const trajectorySamples = Math.min(
		generation.samplesPerTrajectory,
		generation.sampleCount -
			row.trajectoryIndex * generation.samplesPerTrajectory,
	);
	assertInteger(
		sampleIndex,
		"sample index within trajectory",
		0,
		trajectorySamples - 1,
	);
	assertInteger(row.target, "target", 0, contract.numClasses - 1);
	if (
		!Array.isArray(row.input) ||
		row.input.length !== contract.inputShape[0]
	) {
		throw new Error(
			`Input must be a flat vector with ${contract.inputShape[0]} values.`,
		);
	}
	for (let index = 0; index < row.input.length; index++) {
		if (!Number.isFinite(row.input[index]))
			throw new Error(
				`Input value ${index} must be a finite number.`,
			);
	}
}

export function serializeDatasetExample(
	value: unknown,
	contract: DatasetContract,
): string {
	validateDatasetExample(value, contract);
	// Construct an allowlisted record rather than serializing simulator/agent objects.
	return (
		JSON.stringify({
			simulationSeed: value.simulationSeed,
			trajectoryIndex: value.trajectoryIndex,
			step: value.step,
			input: Array.from(value.input),
			target: value.target,
		}) + "\n"
	);
}
