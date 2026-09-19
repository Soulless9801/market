import type { FeatureBuilder, FeatureNormalizer, Model, SeededRandom } from "@/simulation";
import { MLP, ConfigManager, ModelManager, FeatureManager, NormalizerManager, MLPFeatureBuilder, SIDE_ACTIONS, CNNFeatureBuilder } from "@/simulation";

import side_data from "@models/side_model_mlp.json";
import side_norm from "@datasets/side_normalizer_mlp.json";

import test_data from "@models/side_model_cnn.json";
import test_norm from "@datasets/side_normalizer_cnn.json";

export function createSideResolver(random: SeededRandom): Model {
    
    const side_json = side_data;
    
    const config = ConfigManager.build("mlp", MLPFeatureBuilder.featureCount, SIDE_ACTIONS.length);

    const model = ModelManager.build("mlp", config, random);

    model.fromJSON(JSON.stringify(side_json));

    return model;
}

export function createTestResolver(random: SeededRandom): Model {
    
    const test_json = test_data;
    
    const config = ConfigManager.build("cnn", CNNFeatureBuilder.featureCount, SIDE_ACTIONS.length);

    const model = ModelManager.build("cnn", config, random);

    model.fromJSON(JSON.stringify(test_json));

    return model;
}

export function createTestBuilder(): FeatureBuilder {
    const builder = FeatureManager.create("cnn");

    return builder;
}

export function createTestNormalizer(): FeatureNormalizer {

    const normalizer_json = test_norm;
    
    const normalizer = NormalizerManager.getNormalizer("gaussian", null);

    normalizer.fromJSON(JSON.stringify(normalizer_json));

    return normalizer;
}

export function createSideBuilder(): FeatureBuilder {
    
    const builder = FeatureManager.create("mlp");

    return builder;
}

export function createSideNormalizer(): FeatureNormalizer {

    const normalizer_json = side_norm;
    
    const normalizer = NormalizerManager.getNormalizer("gaussian", null);

    normalizer.fromJSON(JSON.stringify(normalizer_json));

    return normalizer;
}

export function createPriceResolver(random: SeededRandom): Model {
    
    const config = {
        kind: "mlp",
        layers: [4, 16, 16, 3]
    };

    const model = new MLP(config, random);

    return model;
}