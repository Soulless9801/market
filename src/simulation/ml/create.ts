import type { FeatureBuilder, Model, SeededRandom } from "@/simulation";
import { MLP, ModelManager, FeatureManager, FeatureNormalizer } from "@/simulation";

import side_data from "@models/side_model_mlp.json";
import side_norm from "@datasets/side_normalizer_mlp.json";

export function createSideResolver(random: SeededRandom): Model {
    
    const side_json = side_data;
    
    const architecture = side_json.architecture;

    const model = ModelManager.build("mlp", architecture, random);

    if (!model) {
        throw new Error("No model builder registered for model: mlp");
    }

    model.fromJSON(JSON.stringify(side_json));

    return model;
}

export function createSideBuilder(): FeatureBuilder {
    
    const builder = FeatureManager.create("mlp");

    if (!builder) {
        throw new Error("No feature builder registered for model: mlp");
    }

    return builder;
}

export function createSideNormalizer(): FeatureNormalizer {

    const normalizer_json = side_norm;
    
    const normalizer = FeatureNormalizer.fromJSON(JSON.stringify(normalizer_json));

    return normalizer;
}

export function createPriceResolver(random: SeededRandom): Model {
    
    const architecture = [4, 16, 16, 3];

    const model = new MLP(architecture, random);

    return model;
}