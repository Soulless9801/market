import { describe, it, expect } from "vitest";
import { generate } from '../src/simulation';
    
// import { writeFile } from 'fs/promises';

// async function writeToFile(data: string, filePath: string): Promise<void> {
//     try {
//         await writeFile(filePath, data, 'utf-8');
//         console.log('File written successfully.');
//     } catch (error) {
//         console.error('Error writing file:', error);
//     }
// }

describe("dataset generation", () => {
    it("should generate the correct number of training examples", () => {
        const examples = generate({ seed: 42, offset: 42, gap: 3, num: 100 });

        expect(examples).toHaveLength(100);

        // console.log(examples);

        for (const example of examples) {
            expect(["SELL", "HOLD", "BUY"]).toContain(example.label);
        }
    });
    it("should generate consistent datasets for the same seed", () => {
        const examples1 = generate({ seed: 42, offset: 42, gap: 3, num: 100 });
        const examples2 = generate({ seed: 42, offset: 42, gap: 3, num: 100 });

        expect(examples1).toEqual(examples2);
    });
});