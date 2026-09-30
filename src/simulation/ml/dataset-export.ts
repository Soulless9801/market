// Node-only module: deliberately excluded from the browser-facing ML barrel.
import { createHash } from "node:crypto";
import { mkdir, open, rename, rm, rmdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import {
	createDatasetContract,
	serializeDatasetExample,
	validateDatasetExample,
} from "./dataset-contract";
import type { DatasetMetadata, DatasetOptions } from "./dataset-contract";
import { generateDataset } from "./dataset-generation";

/** Write into a new directory; existing exports are never overwritten. */
export async function exportDataset(
	outputDirectory: string,
	options: DatasetOptions,
	examples?: Iterable<unknown> | AsyncIterable<unknown>,
): Promise<DatasetMetadata> {
	const contract = createDatasetContract(options);
	const settings = contract.generation;
	const output = resolve(outputDirectory);
	await mkdir(dirname(output), { recursive: true });
	await mkdir(output); // Exclusive reservation also prevents concurrent exporters.
	// Readers treat metadata.json as the completion marker. Stage both files in the
	// reserved directory, validate every row, then publish metadata last.
	const dataTemp = join(output, ".dataset.jsonl.tmp");
	const metadataTemp = join(output, ".metadata.json.tmp");
	const dataFile = join(output, "dataset.jsonl");
	const metadataFile = join(output, "metadata.json");
	try {
		const file = await open(dataTemp, "wx");
		const hash = createHash("sha256");
		const classCounts = { BUY: 0, SELL: 0, HOLD: 0 };
		let sampleCount = 0;
		try {
			for await (const example of examples ??
				generateDataset(settings)) {
				validateDatasetExample(example, contract);
				const trajectory = Math.floor(
					sampleCount /
						contract.generation
							.samplesPerTrajectory,
				);
				const step =
					settings.warmupSteps +
					(sampleCount %
						settings.samplesPerTrajectory) *
						settings.horizonSteps;
				if (
					sampleCount >= settings.sampleCount ||
					example.trajectoryIndex !==
						trajectory ||
					example.step !== step
				) {
					throw new Error(
						`Example ${sampleCount} violates the declared row count or order.`,
					);
				}
				const line = serializeDatasetExample(
					example,
					contract,
				);
				await file.writeFile(line, "utf8"); // Await every write: bounded memory/backpressure.
				hash.update(line, "utf8");
				classCounts[
					contract.classNames[example.target]
				]++;
				sampleCount++;
			}
		} finally {
			await file.close();
		}
		if (sampleCount !== settings.sampleCount)
			throw new Error(
				`Expected ${settings.sampleCount} examples, received ${sampleCount}.`,
			);
		const metadata: DatasetMetadata = {
			...contract,
			datasetFile: "dataset.jsonl",
			sampleCount,
			classCounts,
			sha256: hash.digest("hex"),
		};
		await writeFile(
			metadataTemp,
			JSON.stringify(metadata, null, 2) + "\n",
			{ encoding: "utf8", flag: "wx" },
		);
		await rename(dataTemp, dataFile);
		// Metadata is the completion marker. A reader should also verify the checksum.
		await rename(metadataTemp, metadataFile);
		return metadata;
	} catch (error) {
		// Remove only our known files. Never recursively remove unrelated user content.
		await Promise.allSettled(
			[dataTemp, metadataTemp, dataFile, metadataFile].map(
				(path) => rm(path, { force: true }),
			),
		);
		await rmdir(output).catch(() => {});
		throw error;
	}
}
