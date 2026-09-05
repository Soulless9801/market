import type { Model } from "./models";
import type { SeededRandom } from "@/simulation";
import { MLP } from "./models";

import side_data from "@/side_model.json" with { type: "json" };

export function createSideResolver(random: SeededRandom): Model {
    
    const side_json = side_data;
    
    const architecture = side_json.architecture;

    const model = new MLP(architecture, random);

    model.fromJSON(JSON.stringify(side_json));

    return model;
}

export function createPriceResolver(random: SeededRandom): Model {
    
    const architecture = [4, 16, 16, 3];

    const model = new MLP(architecture, random);

    return model;
}