import { describe, it, expect } from "vitest";
import { MLP } from "../simulation/ml";


describe("MLP", () => {
    it("should predict output for given input", () => {
        const model = new MLP();

        const output = model.predict([
            0.01,   // short return
            -0.02,  // long return
            0.01,   // spread
            0.35,   // imbalance
        ]);

        expect(output).toHaveLength(3);
    });
});