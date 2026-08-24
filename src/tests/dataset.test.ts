import { describe, it, expect } from "vitest";
import { generate } from '../simulation';
import { writeToFile } from '../simulation';

describe("dataset generation", () => {
    it("should generate the correct number of training examples", () => {
        const examples = generate({ seed: 42, offset: 42, gap: 3, num: 100 });

        expect(examples).toHaveLength(100);

        // console.log(examples);

        for (const example of examples) {
            expect(example.features).toHaveLength(4);
            expect(["SELL", "HOLD", "BUY"]).toContain(example.label);
        }
    });
    it("should write generated dataset to a file", async () => {
        const examples = generate({ seed: 42, offset: 42, gap: 3, num: 100 });
        const dataString = JSON.stringify(examples, null, 2);
        const filePath = "dataset.json";

        await writeToFile(dataString, filePath);
    });
});