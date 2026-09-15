import { describe, it, expect } from "vitest";
import { MLP, SeededRandom } from "@/simulation";
import { SIDE_ACTIONS as ACTIONS } from '@/simulation';

const seed = 42;

function createSideModel(inp: number): MLP {
    return new MLP([inp, inp * 4, inp * 2, inp, ACTIONS.length], new SeededRandom(seed));
}

describe("MLP", () => {
    it("should predict output for given input", () => {

        const model = createSideModel(4);

        const output = model.predict([
            0.01,   // short return
            -0.02,  // long return
            0.01,   // spread
            0.35,   // imbalance
        ]);

        expect(output).toHaveLength(ACTIONS.length);
    });
});