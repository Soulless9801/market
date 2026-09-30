import { LOCAL_INFERENCE_PATH, type LocalModelDescriptor } from "../local-inference";
import { validateSideModelMetadata } from "@/simulation/ml/model-binding";
import { SIDE_ACTIONS } from "@/simulation/ml/types";
import type { PredictiveModel } from "@/simulation/ml/predictive-model";

async function requestJson(path: string, body?: unknown): Promise<unknown> {
	const response = await fetch(`${LOCAL_INFERENCE_PATH}/${path}`, {
		method: body === undefined ? "GET" : "POST",
		headers: body === undefined ? undefined : { "Content-Type": "application/json" },
		body: body === undefined ? undefined : JSON.stringify(body),
		signal: AbortSignal.timeout(35_000),
	});
	const result = await response.json();
	if (!response.ok) throw new Error(result.error ?? `Local inference failed (${response.status})`);
	return result;
}

/** Validate the checkpoint's public feature contract before adding an ML participant. */
export async function loadLocalModel(): Promise<{ alias: string; featureLayout: string; model: PredictiveModel }> {
	const descriptor = await requestJson("model") as LocalModelDescriptor;
	if (typeof descriptor.alias !== "string" || !descriptor.alias) throw new Error("Missing local model alias");
	const featureLayout = validateSideModelMetadata(descriptor.metadata);
	return {
		alias: descriptor.alias,
		featureLayout,
		model: {
			async predict(input) {
				const result = await requestJson("predict", { input }) as { prediction?: unknown };
				if (!Array.isArray(result.prediction) || result.prediction.length !== SIDE_ACTIONS.length ||
					result.prediction.some(value => typeof value !== "number" || !Number.isFinite(value))) {
					throw new Error("Invalid prediction from local Python model");
				}
				return result.prediction;
			},
		},
	};
}
