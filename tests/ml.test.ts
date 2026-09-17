import { describe, it, expect } from "vitest";
import { MLP, SeededRandom } from "@/simulation";
import { SIDE_ACTIONS as ACTIONS } from '@/simulation';

const seed = 42;

function createMLPModel(inp: number): MLP {
    return new MLP([inp, inp * 4, inp * 2, inp, ACTIONS.length], new SeededRandom(seed));
}

describe("Models", () => {
    it("MLP should predict output for given input", () => {

        const n = 30;

        // test feature length from 
        for (let i = 1; i <= n; i++) {
            const feature = Array.from({ length: i }, () => Math.random());
            const model = createMLPModel(i);

            const output = model.predict(feature);

            expect(output.length).toBe(ACTIONS.length);
        }
    });
});