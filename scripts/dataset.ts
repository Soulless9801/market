import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import {
	DEFAULT_DATASET_OPTIONS,
	createDatasetContract,
} from "../src/simulation/ml/dataset-contract";
import type {
	DatasetModel,
	DatasetOptions,
} from "../src/simulation/ml/dataset-contract";
import { generateDataset } from "../src/simulation/ml/dataset-generation";
import { exportDataset } from "../src/simulation/ml/dataset-export";
import type { TrainingExample } from "../src/simulation/ml/data";

// Retain the old raw-example helper for callers; large exports use the lazy generator.
export type { TrainingExample } from "../src/simulation/ml/data";
export function generateSideDataset(options: {
	model: string;
	seed: number;
	offset: number;
	gap: number;
	num: number;
}): TrainingExample[] {
	const settings = {
		...DEFAULT_DATASET_OPTIONS,
		model: options.model as DatasetModel,
		seed: options.seed,
		warmupSteps: options.offset,
		horizonSteps: options.gap,
		sampleCount: options.num,
	};
	const contract = createDatasetContract(settings);
	return Array.from(generateDataset(settings), (example) => ({
		features: example.input,
		label: contract.classNames[example.target],
	}));
}

export async function main(args = process.argv.slice(2)): Promise<void> {
	const { values, positionals } = parseArgs({
		args,
		allowPositionals: true,
		options: {
			help: { type: "boolean", short: "h" },
			output: { type: "string" },
			seed: { type: "string" },
			samples: { type: "string" },
			trajectories: { type: "string" },
			warmup: { type: "string" },
			horizon: { type: "string" },
		},
	});
	if (values.help) {
		console.log(`Usage: npm run dataset -- <mlp|cnn> [options]
  --output PATH     New output directory (datasets/training/<model>)
  --seed N          Base uint32 simulation seed (42)
  --samples N       Number of samples (10000)
  --trajectories N  Requested independent trajectories (20)
  --warmup N        Completed steps before sampling (42; MLP minimum 20)
  --horizon N       Label horizon and sample spacing in steps (10)

Writes raw public features to dataset.jsonl and a versioned metadata.json.
Existing output directories are rejected. Legacy dataset/scaler files are unchanged.`);
		return;
	}
	const model = positionals[0];
	if (positionals.length !== 1 || (model !== "mlp" && model !== "cnn")) {
		throw new Error(
			"Specify one feature layout: mlp or cnn. Use --help for options.",
		);
	}
	const integerOption = (
		value: string | undefined,
		fallback: number,
	): number => {
		if (value === undefined) return fallback;
		if (
			!/^\d+$/.test(value) ||
			!Number.isSafeInteger(Number(value))
		) {
			throw new Error(
				`Expected a non-negative safe integer, received ${JSON.stringify(value)}.`,
			);
		}
		return Number(value);
	};
	const options: DatasetOptions = {
		model,
		seed: integerOption(values.seed, DEFAULT_DATASET_OPTIONS.seed),
		sampleCount: integerOption(
			values.samples,
			DEFAULT_DATASET_OPTIONS.sampleCount,
		),
		trajectoryCount: integerOption(
			values.trajectories,
			DEFAULT_DATASET_OPTIONS.trajectoryCount,
		),
		warmupSteps: integerOption(
			values.warmup,
			DEFAULT_DATASET_OPTIONS.warmupSteps,
		),
		horizonSteps: integerOption(
			values.horizon,
			DEFAULT_DATASET_OPTIONS.horizonSteps,
		),
	};
	const output = values.output ?? `datasets/training/${model}`;
	if (!output.trim())
		throw new Error("Output directory cannot be empty.");
	const metadata = await exportDataset(output, options);
	console.log(
		`Exported ${metadata.sampleCount} ${model} samples to ${resolve(output)} (schema v${metadata.schemaVersion}, input [${metadata.inputShape}]).`,
	);
}

if (
	process.argv[1] &&
	resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
	main().catch((error) => {
		console.error("Dataset export failed:", error);
		process.exitCode = 1;
	});
}
