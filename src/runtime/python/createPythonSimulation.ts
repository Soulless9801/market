import { buildAgents } from "@/simulation/agents/TraderAgents";
import { Simulator } from "@/simulation/simulator/Simulator";
import { validateSideModelMetadata } from "@/simulation/ml/model-binding";
import {
	PythonInferenceClient,
	type PythonInferenceOptions,
} from "./PythonInferenceClient";

export interface PythonSimulationOptions {
	python: PythonInferenceOptions;
	modelAlias: string;
	seed?: number;
	referencePrice?: number;
}

/** Own one Python process for the simulation's lifetime; callers must await shutdown in finally. */
export async function createPythonSimulation(options: PythonSimulationOptions) {
	const client = new PythonInferenceClient(options.python);
	try {
		// Bind by the checkpoint's public feature contract, not by its deployment alias.
		// A mismatch must fail before the simulator submits its first order.
		const featureLayout = validateSideModelMetadata(
			await client.getMetadata(options.modelAlias),
		);
		const referencePrice = options.referencePrice ?? 100;
		const simulator = new Simulator({
			referencePrice,
			agents: buildAgents(
				options.seed ?? 15,
				referencePrice,
				{
					featureLayout,
					model: client.model(options.modelAlias),
				},
			),
		});
		return { simulator, shutdown: () => client.shutdown() };
	} catch (error) {
		await client.shutdown();
		throw error;
	}
}
