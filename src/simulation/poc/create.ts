import type { FeatureBuilder } from "@/simulation/ml/feature-generation";
import { FeatureManager } from "@/simulation/ml/feature-generation";
import type { FeatureNormalizer } from "./norm";
import { NormalizerManager } from "./norm";
import type { Model, ModelConfig } from "./models";
import { ModelManager } from "./models";
import type { SeededRandom } from "@/simulation/agents";
import { SIDE_ACTIONS } from "@/simulation/ml/types";
import { DEFAULT_SIDE_MODEL, DEFAULT_PRICE_MODEL } from "./runtime-presets";
import type { SideModelPreset } from "./runtime-presets";

export function createSideBuilder(
	preset: SideModelPreset = DEFAULT_SIDE_MODEL,
): FeatureBuilder {
	return FeatureManager.create(preset.featureLayout);
}
export function createSideNormalizer(
	preset: SideModelPreset = DEFAULT_SIDE_MODEL,
): FeatureNormalizer {
	const normalizer = NormalizerManager.getNormalizer(
		preset.normalizer,
		null,
	);
	normalizer.fromJSON(preset.normalizerJSON);
	return normalizer;
}
export function createSideResolver(
	builder: FeatureBuilder,
	random: SeededRandom,
	preset: SideModelPreset = DEFAULT_SIDE_MODEL,
): Model {
	const saved = JSON.parse(preset.modelJSON) as {
		architecture?: ModelConfig;
	};
	if (!saved.architecture || typeof saved.architecture.kind !== "string")
		throw new Error(
			"Checkpoint is missing its model architecture.",
		);
	// Saved architecture is authoritative, independent of current training defaults.
	const model = ModelManager.build(
		saved.architecture.kind,
		saved.architecture,
		random,
	);
	model.fromJSON(preset.modelJSON);
	const output = model.predict(new Array(builder.featureCount).fill(0));
	if (
		output.length !== SIDE_ACTIONS.length ||
		!output.every(Number.isFinite)
	)
		throw new Error(
			"Side model must produce one finite score per action.",
		);
	return model;
}
export function createPriceResolver(
	random: SeededRandom,
	config: ModelConfig = DEFAULT_PRICE_MODEL,
): Model {
	return ModelManager.build(config.kind, config, random);
}
