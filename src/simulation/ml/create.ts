import type { FeatureBuilder, FeatureNormalizer, Model, SeededRandom } from "@/simulation";
import { MLP, ConfigManager, ModelManager, FeatureManager, NormalizerManager, SIDE_ACTIONS } from "@/simulation";

import side_data from "@models/side_model_mlp.json";
import side_norm from "@datasets/side_normalizer_mlp.json";

const modelStr = "mlp";

export function createSideBuilder(): FeatureBuilder {
    
    const builder = FeatureManager.create(modelStr);

    return builder;
}

export function createSideNormalizer(): FeatureNormalizer {

    const normalizer_json = side_norm;
    
    const normalizer = NormalizerManager.getNormalizer("gaussian", null);

    normalizer.fromJSON(JSON.stringify(normalizer_json));

    return normalizer;
}

export function createSideResolver(builder: FeatureBuilder, random: SeededRandom): Model {
    
    const side_json = side_data;
    
    const config = ConfigManager.build(modelStr, builder.featureCount, SIDE_ACTIONS.length);

    const model = ModelManager.build(modelStr, config, random);

    model.fromJSON(JSON.stringify(side_json));

    return model;
}

export function createPriceResolver(random: SeededRandom): Model {
    
    const config = {
        kind: "mlp",
        layers: [4, 16, 16, 3]
    };

    const model = new MLP(config, random);

    return model;
}