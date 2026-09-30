import { Simulator } from "@/simulation/simulator/Simulator";
import { buildDefaultAgents } from "@/simulation/agents/TraderAgents";
import { FeatureManager } from "./feature-generation";
import {
	classifyMidPriceChange,
	createDatasetContract,
	trajectorySeed,
	validateDatasetExample,
} from "./dataset-contract";
import type { DatasetExample, DatasetOptions } from "./dataset-contract";

/** Lazy generation without retaining the dataset. Capture features before advancing to labels. */
export function* generateDataset(
	options: DatasetOptions,
): Generator<DatasetExample, undefined, unknown> {
	const contract = createDatasetContract(options);
	const settings = contract.generation;
	const builder = FeatureManager.create(settings.model);
	const { samplesPerTrajectory, actualTrajectoryCount } = settings;
	let produced = 0;
	for (
		let trajectoryIndex = 0;
		trajectoryIndex < actualTrajectoryCount;
		trajectoryIndex++
	) {
		const simulationSeed = trajectorySeed(
			settings.seed,
			trajectoryIndex,
		);
		const simulator = new Simulator({
			agents: buildDefaultAgents(simulationSeed),
			referencePrice: 100,
		});
		// Recreate all agents per trajectory so RNG state cannot bleed between seeds.
		while (simulator.getClock() < settings.warmupSteps)
			simulator.runStep();
		for (
			let index = 0;
			index < samplesPerTrajectory &&
			produced < settings.sampleCount;
			index++
		) {
			const context = simulator.getObservableContext(
				contract.features.tradeHistoryLimit,
				contract.features.priceHistoryLimit,
				contract.features.bookDepth,
			);

			const step = simulator.getClock();
			const input = [...builder.build(context)];
			// Freeze today's features before stepping into the future to compute the label.
			// A feature builder never sees the future price used below.
			for (
				let ahead = 0;
				ahead < settings.horizonSteps;
				ahead++
			)
				simulator.runStep();
			// Only the future midprice is used for the target, never for the input.
			const futureMidPrice =
				simulator.getObservableContext().midPrice;
			const label = classifyMidPriceChange(
				context.midPrice,
				futureMidPrice,
			);
			const example = {
				simulationSeed,
				trajectoryIndex,
				step,
				input,
				target: contract.classNames.indexOf(label),
			};
			validateDatasetExample(example, contract);
			yield example;
			produced++;
		}
	}
	return undefined;
}
