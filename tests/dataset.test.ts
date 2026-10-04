import { createHash } from "node:crypto";
import {
	mkdtemp,
	readFile,
	readdir,
	rm,
	mkdir,
	writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
	classifyMidPriceChange,
	createDatasetContract,
	DEFAULT_DATASET_OPTIONS,
	serializeDatasetExample,
	trajectorySeed,
	validateDatasetExample,
} from "@/simulation/ml/dataset-contract";
import type {
	DatasetExample,
	DatasetMetadata,
	DatasetModel,
	DatasetOptions,
} from "@/simulation/ml/dataset-contract";
import { generateDataset } from "@/simulation/ml/dataset-generation";
import { exportDataset } from "@/simulation/ml/dataset-export";
import { FeatureManager } from "@/simulation/ml/feature-generation";
import { Simulator } from "@/simulation/simulator/Simulator";
import { buildDefaultAgents } from "@/simulation/agents/TraderAgents";

const options = (model: DatasetModel = "mlp"): DatasetOptions => ({
	...DEFAULT_DATASET_OPTIONS,
	model,
	sampleCount: 5,
	trajectoryCount: 2,
});
const directories: string[] = [];
async function workspace() {
	const directory = await mkdtemp(join(tmpdir(), "market-dataset-test-"));
	directories.push(directory);
	return directory;
}
afterEach(async () => {
	vi.restoreAllMocks();
	await Promise.all(
		directories
			.splice(0)
			.map((directory) =>
				rm(directory, { recursive: true, force: true }),
			),
	);
});

function validExample(settings = options()): DatasetExample {
	return {
		simulationSeed: settings.seed,
		trajectoryIndex: 0,
		step: settings.warmupSteps,
		input: Array(settings.model === "mlp" ? 40 : 30).fill(0),
		target: 0,
	};
}

describe("Dataset contract", () => {
	it.each(["mlp", "cnn"] as const)(
		"describes the actual %s input and fixed class order",
		(model) => {
			const contract = createDatasetContract(options(model));
			expect(contract.schemaVersion).toBe(1);
			expect(contract.inputShape).toEqual([
				model === "mlp" ? 40 : 30,
			]);
			expect(contract.features.names).toHaveLength(
				contract.inputShape[0],
			);
			expect(new Set(contract.features.names).size).toBe(
				contract.inputShape[0],
			);
			expect(contract.classNames).toEqual([
				"BUY",
				"SELL",
				"HOLD",
			]);
			expect(contract.numClasses).toBe(3);
			expect(contract.targetEncoding).toBe("class-index");
			expect(contract.features.normalization).toBe("none");
			expect(contract.label).toMatchObject({
				threshold: 0.01,
				thresholdUnit: "absolute-price",
				horizonSteps: 10,
			});
			expect(contract.generation).toMatchObject({
				samplesPerTrajectory: 3,
				actualTrajectoryCount: 2,
			});
			const row = validExample(options(model));
			const encoded = serializeDatasetExample(row, contract);
			expect(encoded.endsWith("\n")).toBe(true);
			expect(JSON.parse(encoded)).toEqual(row);
			expect(Object.keys(JSON.parse(encoded))).toEqual([
				"simulationSeed",
				"trajectoryIndex",
				"step",
				"input",
				"target",
			]);
		},
	);

	it("preserves strict threshold comparisons and HOLD at both boundaries", () => {
		for (const [change, expected] of [
			[0.010001, "BUY"],
			[-0.010001, "SELL"],
			[0.01, "HOLD"],
			[-0.01, "HOLD"],
			[0, "HOLD"],
		] as const) {
			expect(classifyMidPriceChange(0, change)).toBe(
				expected,
			);
		}
		expect(() => classifyMidPriceChange(NaN, 1)).toThrow(/finite/);
		expect(() => classifyMidPriceChange(1, Infinity)).toThrow(
			/finite/,
		);
		expect(() =>
			classifyMidPriceChange(
				-Number.MAX_VALUE,
				Number.MAX_VALUE,
			),
		).toThrow(/finite/);
	});

	it("does not copy private or unrelated configuration into metadata", () => {
		const settings = {
			...options(),
			portfolio: { cash: 100000 },
			participantId: "private",
		};
		const contract = createDatasetContract(settings);
		expect(JSON.stringify(contract)).not.toContain("portfolio");
		expect(JSON.stringify(contract)).not.toContain("participantId");
	});

	it("rejects malformed vectors, labels, provenance and extra private fields before serialization", () => {
		const row = validExample();
		const contract = createDatasetContract(options());
		const malformed: unknown[] = [
			null,
			[],
			{},
			{ ...row, portfolio: { cash: 100000 } },
			{ ...row, participantId: "private" },
			{ ...row, input: row.input.slice(1) },
			{ ...row, input: [...row.input, 1] },
			{ ...row, input: [row.input] },
			{ ...row, input: new Array(40) },
			...[NaN, Infinity, -Infinity, "1", null, undefined].map(
				(value) => ({
					...row,
					input: [value, ...row.input.slice(1)],
				}),
			),
			...[-1, 3, 1.5, NaN, "BUY"].map((target) => ({
				...row,
				target,
			})),
			{ ...row, simulationSeed: 43 },
			{ ...row, simulationSeed: -1 },
			{ ...row, trajectoryIndex: 2 },
			{ ...row, trajectoryIndex: 0.5 },
			{ ...row, step: 41 },
			{ ...row, step: 43 },
			{ ...row, step: 72 },
		];
		for (const example of malformed)
			expect(() =>
				serializeDatasetExample(example, contract),
			).toThrow();
	});

	it("validates generation settings, including MLP warmup, without inventing padded features", () => {
		const invalid = [
			{ model: "unknown" },
			{ seed: -1 },
			{ seed: 2 ** 32 },
			{ seed: 1.5 },
			{ sampleCount: 0 },
			{ sampleCount: Infinity },
			{ sampleCount: Number.MAX_SAFE_INTEGER + 1 },
			{ trajectoryCount: 0 },
			{ trajectoryCount: 2 ** 32 + 1 },
			{ warmupSteps: 19 },
			{ horizonSteps: 0 },
			{ horizonSteps: NaN },
			{ horizonSteps: Number.MAX_SAFE_INTEGER },
		];
		for (const overrides of invalid)
			expect(() =>
				createDatasetContract({
					...options(),
					...overrides,
				} as DatasetOptions),
			).toThrow();
		expect(() =>
			createDatasetContract({
				...options(),
				warmupSteps: 20,
			}),
		).not.toThrow();
		expect(() =>
			createDatasetContract({
				...options("cnn"),
				warmupSteps: 0,
			}),
		).not.toThrow();
	});
});

describe("Dataset generation", () => {
	it.each(["mlp", "cnn"] as const)(
		"reproduces %s samples and labels from public observations before the future horizon",
		(model) => {
			const settings = options(model);
			const rows = [...generateDataset(settings)];
			expect(rows).toEqual([...generateDataset(settings)]);
			expect(rows).toHaveLength(5);
			expect(
				rows.map((row) => [
					row.trajectoryIndex,
					row.step,
					row.simulationSeed,
				]),
			).toEqual([
				[0, 42, 42],
				[0, 52, 42],
				[0, 62, 42],
				[1, 42, 10015],
				[1, 52, 10015],
			]);
			const builder = FeatureManager.create(model);
			const classes = ["BUY", "SELL", "HOLD"];
			for (const index of [0, 1]) {
				const simulator = new Simulator({
					agents: buildDefaultAgents(
						42 + index * 9973,
					),
				});
				simulator.runSteps(42);
				for (const row of rows.filter(
					(example) =>
						example.trajectoryIndex ===
						index,
				)) {
					const before =
						simulator.getObservableContext(
							20,
							20,
						);
					const expectedInput =
						builder.build(before);
					expect(row.input).toEqual(
						expectedInput,
					);
					expect(row.step).toBe(
						simulator.getClock(),
					);
					simulator.runSteps(10);
					const change =
						simulator.getObservableContext()
							.midPrice -
						before.midPrice;
					expect(row.target).toBe(
						classes.indexOf(
							change > 0.01
								? "BUY"
								: change < -0.01
									? "SELL"
									: "HOLD",
						),
					);
					expect(row.input).toEqual(
						expectedInput,
					);
				}
			}
			expect(rows).not.toEqual([
				...generateDataset({ ...settings, seed: 43 }),
			]);
		},
	);

	it("preserves CNN current-price padding and permits its existing cold-start behavior", () => {
		const rows = [
			...generateDataset({
				...options("cnn"),
				warmupSteps: 0,
				sampleCount: 1,
			}),
		];
		expect(rows[0].input).toEqual(Array(30).fill(100));
		const simulator = new Simulator({
			agents: buildDefaultAgents(42),
		});
		simulator.runSteps(42);
		const observation = simulator.getObservableContext(20, 20);
		const warmRow = generateDataset(options("cnn")).next().value!;
		expect(warmRow.input).toEqual([
			...Array(10).fill(observation.midPrice),
			...observation.recentMidPriceSeries,
		]);
	});

	it("records wrapped trajectory seeds consistently with the existing seeded simulator", () => {
		const settings = {
			...options(),
			seed: 0xffffffff,
			sampleCount: 2,
			trajectoryCount: 2,
		};
		const rows = [...generateDataset(settings)];
		expect(rows.map((row) => row.simulationSeed)).toEqual([
			0xffffffff, 9972,
		]);
		expect(trajectorySeed(0xffffffff, 1)).toBe(9972);
		const simulator = new Simulator({
			agents: buildDefaultAgents(0xffffffff + 9973),
		});
		simulator.runSteps(42);
		expect(rows[1].input).toEqual(
			FeatureManager.create("mlp").build(
				simulator.getObservableContext(),
			),
		);
	});

	it("is lazy, retains legacy trajectory partitioning, and stops after the requested samples", () => {
		const settings = {
			...options(),
			sampleCount: 21,
			trajectoryCount: 20,
		};
		expect(
			createDatasetContract(settings).generation,
		).toMatchObject({
			samplesPerTrajectory: 2,
			actualTrajectoryCount: 11,
		});
		const build = vi.spyOn(FeatureManager, "create");
		const iterator = generateDataset({
			...options(),
			sampleCount: 1000000,
		});
		expect(build).not.toHaveBeenCalled();
		expect(iterator.next().value!.input).toHaveLength(40);
		iterator.return(undefined);
		expect(build).toHaveBeenCalledTimes(1);
	});
});

describe("JSONL export", () => {
	it.each(["mlp", "cnn"] as const)(
		"streams %s rows with exact metadata, counts, checksum and reproducible bytes",
		async (model) => {
			const parent = await workspace();
			const first = join(parent, "first");
			const second = join(parent, "second");
			const settings = options(model);
			const metadata = await exportDataset(first, settings);
			await exportDataset(second, settings);
			const bytes = await readFile(
				join(first, "dataset.jsonl"),
				"utf8",
			);
			expect(bytes).toBe(
				await readFile(
					join(second, "dataset.jsonl"),
					"utf8",
				),
			);
			const metadataText = await readFile(
				join(first, "metadata.json"),
				"utf8",
			);
			expect(metadataText).toBe(
				await readFile(
					join(second, "metadata.json"),
					"utf8",
				),
			);
			const saved = JSON.parse(
				metadataText,
			) as DatasetMetadata;
			expect(saved).toEqual(metadata);
			expect(saved.sampleCount).toBe(5);
			expect(saved.sha256).toBe(
				createHash("sha256")
					.update(bytes)
					.digest("hex"),
			);
			const rows = bytes
				.trimEnd()
				.split("\n")
				.map(
					(line) =>
						JSON.parse(
							line,
						) as DatasetExample,
				);
			expect(rows).toEqual([...generateDataset(settings)]);
			for (const row of rows)
				validateDatasetExample(row, saved);
			const counts = { BUY: 0, SELL: 0, HOLD: 0 };
			rows.forEach(
				(row) => counts[saved.classNames[row.target]]++,
			);
			expect(saved.classCounts).toEqual(counts);
			expect((await readdir(first)).sort()).toEqual([
				"dataset.jsonl",
				"metadata.json",
			]);
		},
	);

	it("rejects invalid rows, wrong counts and duplicate ordering, cleaning up incomplete exports", async () => {
		const parent = await workspace();
		const settings = options();
		const rows = [...generateDataset(settings)];
		const broken = [
			[rows[0], { ...rows[1], input: [1] }],
			rows.slice(0, -1),
			[...rows, rows[0]],
			[rows[0], rows[0], ...rows.slice(2)],
		];
		for (const [index, examples] of broken.entries()) {
			const output = join(parent, `invalid-${index}`);
			await expect(
				exportDataset(output, settings, examples),
			).rejects.toThrow();
			await expect(readdir(output)).rejects.toMatchObject({
				code: "ENOENT",
			});
		}
	});

	it("propagates async producer failures and removes partial output", async () => {
		const output = join(await workspace(), "failed");
		async function* fail() {
			yield validExample();
			throw new Error("producer failed");
		}
		await expect(
			exportDataset(output, options(), fail()),
		).rejects.toThrow("producer failed");
		await expect(readdir(output)).rejects.toMatchObject({
			code: "ENOENT",
		});
	});

	it("leaves existing files untouched and reserves output exclusively for concurrent runs", async () => {
		const parent = await workspace();
		const existing = join(parent, "existing");
		await mkdir(existing);
		await writeFile(join(existing, "dataset.jsonl"), "original");
		await expect(
			exportDataset(existing, options()),
		).rejects.toMatchObject({ code: "EEXIST" });
		expect(
			await readFile(join(existing, "dataset.jsonl"), "utf8"),
		).toBe("original");
		const concurrent = join(parent, "concurrent");
		const attempts = await Promise.allSettled([
			exportDataset(concurrent, options()),
			exportDataset(concurrent, options()),
		]);
		expect(
			attempts.filter(
				(attempt) => attempt.status === "fulfilled",
			),
		).toHaveLength(1);
		expect(
			attempts.filter(
				(attempt) => attempt.status === "rejected",
			),
		).toHaveLength(1);
		expect(
			JSON.parse(
				await readFile(
					join(concurrent, "metadata.json"),
					"utf8",
				),
			).sampleCount,
		).toBe(5);
	});

	it("rejects invalid settings before creating an output directory", async () => {
		const output = join(await workspace(), "invalid-options");
		await expect(
			exportDataset(output, { ...options(), sampleCount: 0 }),
		).rejects.toThrow(/sampleCount/);
		await expect(readdir(output)).rejects.toMatchObject({
			code: "ENOENT",
		});
	});
});

describe("Dataset CLI", () => {
	it("has no import side effects, supports the old raw helper, and exports only when called", async () => {
		const log = vi
			.spyOn(console, "log")
			.mockImplementation(() => {});
		const { main, generateSideDataset } =
			await import("../scripts/dataset");
		expect(log).not.toHaveBeenCalled();
		const legacy = generateSideDataset({
			model: "mlp",
			seed: 42,
			offset: 42,
			gap: 10,
			num: 2,
		});
		const contract = createDatasetContract({
			...options(),
			sampleCount: 2,
			trajectoryCount: 20,
		});
		const expected = [...generateDataset(contract.generation)].map(
			(row) => ({
				features: row.input,
				label: contract.classNames[row.target],
			}),
		);
		expect(legacy).toEqual(expected);
		const output = join(await workspace(), "cli");
		await main([
			"cnn",
			"--samples",
			"3",
			"--trajectories",
			"1",
			"--seed",
			"7",
			"--warmup",
			"25",
			"--horizon",
			"2",
			"--output",
			output,
		]);
		const metadata = JSON.parse(
			await readFile(join(output, "metadata.json"), "utf8"),
		);
		expect(metadata.generation).toMatchObject({
			model: "cnn",
			sampleCount: 3,
			trajectoryCount: 1,
			seed: 7,
			warmupSteps: 25,
			horizonSteps: 2,
		});
		expect(log).toHaveBeenCalledTimes(1);
		await expect(main(["mlp", "--samples", "1.5"])).rejects.toThrow(
			/integer/,
		);
		await expect(main(["unknown"])).rejects.toThrow(/layout/);
	});
});

describe("Order book feature layout", () => {
	const context = (
		recentMidPriceSeries: number[],
		bids: { price: number; quantity: number }[],
		asks: { price: number; quantity: number }[],
	) => ({
		clock: 50,
		midPrice: 100,
		referencePrice: 100,
		spread: 1,
		orderBook: {
			bids: bids.map((level) => ({ ...level, orderCount: 1 })),
			asks: asks.map((level) => ({ ...level, orderCount: 1 })),
		},
		recentTrades: [],
		recentMidPriceSeries,
		orderImbalance: {
			bidVolume: 0,
			askVolume: 0,
			imbalance: 0,
			bidPercent: 0,
			askPercent: 0,
		},
	});

	it("registers a fixed-width layout that needs no trade history", () => {
		const definition = FeatureManager.describe("orderbook");
		expect(definition.names).toHaveLength(80);
		expect(definition).toMatchObject({
			minimumWarmupSteps: 20,
			priceHistoryLimit: 20,
			tradeHistoryLimit: 0,
			bookDepth: 10,
		});
		expect(createDatasetContract(options("orderbook")).inputShape).toEqual([80]);
	});

	it("encodes chronological returns, level offsets, quantities and cumulative depth", () => {
		const prices = Array.from({ length: 20 }, (_, i) => 90 + i * 0.5);
		const input = FeatureManager.create("orderbook").build(
			context(prices, [{ price: 99.5, quantity: 3 }, { price: 99, quantity: 5 }], [{ price: 100.5, quantity: 7 }]),
		);
		expect(input).toHaveLength(80);
		expect(input[0]).toBeCloseTo(Math.log(90.5 / 90));
		expect(input[19]).toBeCloseTo(Math.log(100 / 99.5));
		expect(input.slice(20, 26)).toEqual([
			Math.log(99.5 / 100), Math.log1p(3), Math.log1p(3),
			Math.log(99 / 100), Math.log1p(5), Math.log1p(8),
		]);
		// Missing levels have zero offset and quantity; cumulative depth carries forward.
		expect(input.slice(26, 50)).toEqual(
			Array.from({ length: 8 }, () => [0, 0, Math.log1p(8)]).flat(),
		);
		expect(input.slice(50, 56)).toEqual([
			Math.log(100.5 / 100), Math.log1p(7), Math.log1p(7),
			0, 0, Math.log1p(7),
		]);
		expect(input.every(Number.isFinite)).toBe(true);
	});

	it("left-pads short histories with zero returns instead of changing width", () => {
		const input = FeatureManager.create("orderbook").build(context([99, 99.5], [], []));
		expect(input).toHaveLength(80);
		expect(input.slice(0, 18)).toEqual(Array(18).fill(0));
		expect(input[18]).toBeCloseTo(Math.log(99.5 / 99));
		expect(input[19]).toBeCloseTo(Math.log(100 / 99.5));
	});

	it("exports rows equal to the builder applied to the public observation", () => {
		const settings = options("orderbook");
		const rows = [...generateDataset(settings)];
		expect(rows).toEqual([...generateDataset(settings)]);
		const simulator = new Simulator({ agents: buildDefaultAgents(42) });
		simulator.runSteps(42);
		expect(rows[0].input).toEqual(
			FeatureManager.create("orderbook").build(simulator.getObservableContext(0, 20, 10)),
		);
	});
});
