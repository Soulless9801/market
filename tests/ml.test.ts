import { describe, it, expect } from "vitest";
import { ArchitectureManager, MLP, ModelManager, SeededRandom } from "@/simulation";
import { SIDE_ACTIONS as ACTIONS } from '@/simulation';

const seed = 42;

function createMLPModel(inp: number): MLP {
    const architecture = ArchitectureManager.build('mlp', inp, ACTIONS.length);
    return ModelManager.build('mlp', architecture, new SeededRandom(seed)) as MLP;
}

describe("Models", () => {
    it("MLP should predict output for given input", () => {

        const n = 30;
        const m = 1000;

        // test feature length from 
        for (let i = 1; i <= n; i++) {
            
            const model = createMLPModel(i);

            for (let j = 0; j < m; j++) {
                const feature = Array.from({ length: i }, () => Math.random());
                const output = model.predict(feature);

                expect(output.length).toBe(ACTIONS.length);
            }
        }
    });
});