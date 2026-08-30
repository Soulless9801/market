import type { Model } from "./models";
import type { SeededRandom } from "../agents";
import { MLP } from "./models";

import data from "../../../model.json" with { type: "json" };

export function createSideResolver(random: SeededRandom): Model {
    const architecture = [4, 16, 16, 3];
    const model = new MLP(architecture, random);

    model.fromJSON(JSON.stringify(data));

    return model;
}