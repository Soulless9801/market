import sideModel from "@models/poc/side_model_mlp.json";
import sideNormalizer from "@models/poc/side_normalizer_mlp.json";
import type { ModelConfig } from "./models";

export interface SideModelPreset {
	featureLayout: string;
	normalizer: string;
	normalizerJSON: string;
	modelJSON: string;
}

/** Application composition defaults; shared inference code does not select a model kind. */
export const DEFAULT_SIDE_MODEL: Readonly<SideModelPreset> = Object.freeze({
	featureLayout: "mlp",
	normalizer: "gaussian",
	normalizerJSON: JSON.stringify(sideNormalizer),
	modelJSON: JSON.stringify(sideModel),
});
export const DEFAULT_PRICE_MODEL: ModelConfig & { layers: number[] } = {
	kind: "mlp",
	layers: [4, 16, 16, 3],
};
