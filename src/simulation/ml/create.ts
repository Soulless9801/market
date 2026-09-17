import type { Model, SeededRandom } from "@/simulation";
import { MLP, FeatureNormalizer } from "@/simulation";

import side_data from "@models/side_model_mlp.json";
import side_norm from "@datasets/side_normalizer_mlp.json";

export function createSideResolver(random: SeededRandom): Model {
    
    const side_json = side_data;
    
    const architecture = side_json.architecture;

    const model = new MLP(architecture, random);

    model.fromJSON(JSON.stringify(side_json));

    return model;
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