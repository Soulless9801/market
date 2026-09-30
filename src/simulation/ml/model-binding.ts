import { FeatureManager } from "./feature-generation";
import { SIDE_ACTIONS } from "./types";

/** Refuse mismatched feature order, semantics or output ordering before any market action. */
export function validateSideModelMetadata(metadata: unknown): string {
	// Equal vector lengths alone are insufficient: reordered or redefined features
	// would produce plausible numbers for the wrong inputs. Compare the full meaning.
	if (!metadata || typeof metadata !== "object")
		throw new Error("Missing model metadata");
	const saved = metadata as Record<string, unknown>;
	const features = saved.features as Record<string, unknown> | undefined;
	if (!features || typeof features.kind !== "string")
		throw new Error("Missing feature layout");
	const definition = FeatureManager.describe(features.kind);
	const expected: Record<string, unknown> = {
		version: definition.version,
		names: definition.names,
		normalization: "none",
		historyOrder: definition.historyOrder,
		priceHistoryLimit: definition.priceHistoryLimit,
		tradeHistoryLimit: definition.tradeHistoryLimit,
		bookDepth: definition.bookDepth,
		padding: definition.padding,
		description: definition.description,
	};
	for (const [key, value] of Object.entries(expected)) {
		if (JSON.stringify(features[key]) !== JSON.stringify(value)) {
			throw new Error(
				`Checkpoint feature contract mismatch: ${key}`,
			);
		}
	}
	if (
		JSON.stringify(saved.inputShape) !==
			JSON.stringify([definition.names.length]) ||
		JSON.stringify(saved.classNames) !==
			JSON.stringify(SIDE_ACTIONS) ||
		saved.outputType !== "logits"
	) {
		throw new Error(
			"Checkpoint input shape or action ordering is incompatible",
		);
	}
	return features.kind;
}
