import { LOCAL_INFERENCE_PATH, type LocalModelCatalog, type LocalModelDescriptor } from "../local-inference";
import { validateSideModelMetadata } from "@/simulation/ml/model-binding";
import { SIDE_ACTIONS } from "@/simulation/ml/types";
import type { PredictiveModel } from "@/simulation/ml/predictive-model";

async function requestJson(path: string, body?: unknown): Promise<unknown> {
	const response = await fetch(`${LOCAL_INFERENCE_PATH}/${path}`, {
		method: body === undefined ? "GET" : "POST",
		headers: body === undefined ? undefined : { "Content-Type": "application/json" },
		body: body === undefined ? undefined : JSON.stringify(body),
		// The first metadata request may cold-start Vercel's 60-second Python function.
		signal: AbortSignal.timeout(body === undefined ? 65_000 : 35_000),
	});
	if (!response.headers.get("content-type")?.includes("application/json")) {
		throw new Error("Python inference is not configured on this server");
	}
	const result = await response.json();
	if (!response.ok) throw new Error(result.error ?? `Local inference failed (${response.status})`);
	return result;
}

/** Validate the checkpoint's public feature contract before adding an ML participant. */
export interface LocalModelBinding {
	alias: string;
	featureLayout: string;
	model: PredictiveModel;
}

/** A binding closes over its alias so another selection cannot retarget an in-flight prediction. */
export function bindLocalModel(descriptor: LocalModelDescriptor): LocalModelBinding {
	if (!descriptor || typeof descriptor !== "object") throw new Error("Invalid model descriptor");
	if (typeof descriptor.alias !== "string" || !descriptor.alias) throw new Error("Missing local model alias");
	const alias = descriptor.alias;
	const featureLayout = validateSideModelMetadata(descriptor.metadata);
	return {
		alias,
		featureLayout,
		model: {
			async predict(input) {
				const result = await requestJson("predict", { input, model: alias }) as { prediction?: unknown };
				if (!Array.isArray(result.prediction) || result.prediction.length !== SIDE_ACTIONS.length ||
					result.prediction.some(value => typeof value !== "number" || !Number.isFinite(value))) {
					throw new Error("Invalid prediction from local Python model");
				}
				return result.prediction;
			},
		},
	};
}

export async function loadLocalModel(alias?: string): Promise<LocalModelBinding> {
	const path = alias === undefined ? "model" : `model?alias=${encodeURIComponent(alias)}`;
	const descriptor = await requestJson(path) as LocalModelDescriptor;
	if (alias !== undefined && descriptor?.alias !== alias) throw new Error("Model alias mismatch");
	return bindLocalModel(descriptor);
}

export async function loadLocalModelCatalog(): Promise<LocalModelCatalog> {
	const catalog = await requestJson("models") as LocalModelCatalog;
	if (!catalog || typeof catalog !== "object" || !Array.isArray(catalog.models) || !catalog.models.length)
		throw new Error("Invalid model catalog");
	const aliases = new Set<string>();
	for (const descriptor of catalog.models) {
		bindLocalModel(descriptor); // Validate every feature contract before offering a selection.
		if (aliases.has(descriptor.alias)) throw new Error("Duplicate model alias");
		aliases.add(descriptor.alias);
	}
	if (!aliases.has(catalog.defaultAlias)) throw new Error("Default model is not in the catalog");
	return catalog;
}
