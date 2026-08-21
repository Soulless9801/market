import { describe, it, expect } from "vitest";
import { generate } from '../simulation';

describe("dataset generation", () => {
    it("should generate the correct number of training examples", () => {
        const examples = generate(42, 42, 3, 100);

        expect(examples).toHaveLength(100);

        // console.log(examples);

        for (const example of examples) {
            expect(example.features).toHaveLength(4);
            expect(["SELL", "HOLD", "BUY"]).toContain(example.label);
        }
    });
});