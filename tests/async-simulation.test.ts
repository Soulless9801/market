import { describe, expect, it, vi } from "vitest";
import {
	buildDefaultAgents,
	FeatureManager,
	MLTraderAgent,
	Simulator,
	SIDE_ACTIONS,
} from "@/simulation";
import type {
	AgentSimulatorContext,
	ObservableSimulatorContext,
	PredictiveModel,
} from "@/simulation";
import { validateSideModelMetadata } from "@/simulation/ml/model-binding";

const observations: ObservableSimulatorContext[] = [];
FeatureManager.register("independent-public-layout", {
	version: 1,
	names: ["midprice", "completed_step"],
	minimumWarmupSteps: 0,
	priceHistoryLimit: 2,
	tradeHistoryLimit: 1,
	bookDepth: 3,
	historyOrder: "oldest-to-newest",
	padding: "none",
	description: "Public price probe",
	create: () => ({
		featureCount: 2,
		build(context) {
			observations.push(context);
			return [context.midPrice, context.clock];
		},
	}),
});
function agent(
	model: PredictiveModel,
	featureLayout = "independent-public-layout",
) {
	return new MLTraderAgent("ml", {
		model,
		featureLayout,
		referencePrice: 100,
		spread: 2,
		quantity: 5,
		seed: 42,
	});
}
function deferred() {
	let resolve!: (scores: number[]) => void;
	let reject!: (error: Error) => void;
	const promise = new Promise<number[]>((yes, no) => {
		resolve = yes;
		reject = no;
	});
	return { promise, resolve, reject };
}
function metadata(layout = "independent-public-layout") {
	const definition = FeatureManager.describe(layout);
	const features = {
		version: definition.version,
		names: definition.names,
		priceHistoryLimit: definition.priceHistoryLimit,
		tradeHistoryLimit: definition.tradeHistoryLimit,
		bookDepth: definition.bookDepth,
		historyOrder: definition.historyOrder,
		padding: definition.padding,
		description: definition.description,
	};
	return {
		inputShape: [features.names.length],
		classNames: [...SIDE_ACTIONS],
		outputType: "logits",
		features: { ...features, kind: layout, normalization: "none" },
	};
}

describe("asynchronous simulation", () => {
	it("waits for prediction, consumes public features and submits orders in agent order before committing the step", async () => {
		observations.length = 0;
		const pending = deferred();
		const predict = vi.fn(() => pending.promise);
		const later = vi.fn((context: AgentSimulatorContext) => {
			expect(context.clock).toBe(1);
			expect(simulator.getClock()).toBe(0);
			expect(
				simulator.getOrderBookSnapshot().bids,
			).toHaveLength(1);
			return [];
		});
		const simulator = new Simulator({
			agents: [
				agent({ predict }),
				{ id: "later", step: later },
			],
		});
		expect(() => simulator.runStep()).toThrow("runStepAsync");
		const step = simulator.runStepAsync();
		expect(predict).toHaveBeenCalledExactlyOnceWith([100, 0]);
		expect(observations[0]).not.toHaveProperty("portfolio");
		expect(observations[0].clock).toBe(0);
		expect(simulator.getClock()).toBe(0);
		expect(simulator.getMidPriceHistory()).toEqual([]);
		expect(simulator.getEvents()).toEqual([]);
		expect(later).not.toHaveBeenCalled();
		await expect(simulator.runStepAsync()).rejects.toThrow(
			"pending",
		);
		expect(() => simulator.reset()).toThrow("pending");
		pending.resolve([8, 1, 0]);
		const result = await step;
		expect(later).toHaveBeenCalledTimes(1);
		expect(result.step).toBe(1);
		expect(result.events.map((event) => event.type)).toEqual([
			"agent-step",
			"order-submitted",
			"agent-step",
		]);
		expect(simulator.getMidPriceHistory()).toEqual([100]);
		expect(simulator.getClock()).toBe(1);
	});

	it("preserves synchronous market semantics when the same agents are awaited", async () => {
		for (const seed of [7, 15, 42]) {
			const sync = new Simulator({
				agents: buildDefaultAgents(seed),
			});
			const async = new Simulator({
				agents: buildDefaultAgents(seed),
			});
			for (let step = 0; step < 30; step++) {
				expect(await async.runStepAsync()).toEqual(
					sync.runStep(),
				);
			}
			expect(async.getParticpantPortfolios()).toEqual(
				sync.getParticpantPortfolios(),
			);
			expect(async.getMidPriceHistory()).toEqual(
				sync.getMidPriceHistory(),
			);
		}
	});

	it("derives warmup and raw input shape from the selected feature definition", async () => {
		const predict = vi
			.fn<(input: number[]) => Promise<number[]>>()
			.mockResolvedValue([0, 0, 1]);
		const simulator = new Simulator({
			agents: [agent({ predict }, "mlp")],
		});
		for (
			let index = 0;
			index <
			FeatureManager.describe("mlp").minimumWarmupSteps;
			index++
		)
			await simulator.runStepAsync();
		expect(predict).not.toHaveBeenCalled();
		await simulator.runStepAsync();
		expect(predict).toHaveBeenCalledTimes(1);
		expect(predict.mock.calls[0][0]).toHaveLength(
			FeatureManager.describe("mlp").names.length,
		);
		expect(
			simulator
				.getEvents()
				.filter(
					(event) =>
						event.type ===
						"order-submitted",
				),
		).toEqual([]);
	});

	it.each([
		[9, 1, 0],
		[0, 9, 0],
		[0, 0, 9],
		[1, 1, 1],
	])(
		"keeps BUY SELL HOLD ordering and deterministic tie breaking: %j",
		async (...scores) => {
			const simulator = new Simulator({
				agents: [
					agent({ predict: async () => scores }),
				],
			});
			await simulator.runStepAsync();
			const book = simulator.getOrderBookSnapshot();
			const side =
				SIDE_ACTIONS[
					scores.indexOf(Math.max(...scores))
				];
			expect(book.bids.length).toBe(side === "BUY" ? 1 : 0);
			expect(book.asks.length).toBe(side === "SELL" ? 1 : 0);
		},
	);

	it.each([[], [1, 2], [1, NaN, 3], [1, Infinity, 3]])(
		"fails closed for invalid predictions: %j",
		async (...scores) => {
			const simulator = new Simulator({
				agents: [
					agent({ predict: async () => scores }),
				],
			});
			await expect(simulator.runStepAsync()).rejects.toThrow(
				"finite score",
			);
			expect(simulator.getClock()).toBe(0);
			expect(simulator.getOrderBookSnapshot()).toEqual({
				bids: [],
				asks: [],
			});
			await expect(simulator.runStepAsync()).rejects.toThrow(
				"reset or recreate",
			);
		},
	);

	it("stops on inference failure without running later agents or retrying a partial step", async () => {
		const pending = deferred();
		const later = vi.fn(() => []);
		const simulator = new Simulator({
			agents: [
				agent({ predict: () => pending.promise }),
				{ id: "later", step: later },
			],
		});
		const failed = expect(simulator.runStepAsync()).rejects.toThrow(
			"backend failed",
		);
		pending.reject(new Error("backend failed"));
		await failed;
		expect(later).not.toHaveBeenCalled();
		expect(simulator.getClock()).toBe(0);
		await expect(simulator.runStepAsync()).rejects.toThrow(
			"reset or recreate",
		);
		simulator.reset();
		expect(simulator.getEvents()).toEqual([]);
	});

	it("validates independently named checkpoint feature registrations and rejects semantic/order mismatches", () => {
		expect(validateSideModelMetadata(metadata())).toBe(
			"independent-public-layout",
		);
		for (const changed of [
			{ ...metadata(), inputShape: [40] },
			{ ...metadata(), classNames: ["SELL", "BUY", "HOLD"] },
			{ ...metadata(), outputType: "probabilities" },
			{
				...metadata(),
				features: {
					...metadata().features,
					version: 2,
				},
			},
			{
				...metadata(),
				features: {
					...metadata().features,
					normalization: "gaussian",
				},
			},
			{
				...metadata(),
				features: {
					...metadata().features,
					names: ["private-equity"],
				},
			},
		])
			expect(() =>
				validateSideModelMetadata(changed),
			).toThrow();
	});
});

it("keeps the lock until an unmarked async agent settles if called through the synchronous API", async () => {
	let finish!: (orders: []) => void;
	const pending = new Promise<[]>((resolve) => {
		finish = resolve;
	});
	const simulator = new Simulator({
		agents: [{ id: "unmarked", step: () => pending }],
	});
	expect(() => simulator.runStep()).toThrow("runStepAsync");
	expect(() => simulator.reset()).toThrow("pending");
	finish([]);
	await pending;
	expect(simulator.getClock()).toBe(0);
	expect(() => simulator.runStep()).toThrow("reset or recreate");
	expect(() => simulator.reset()).not.toThrow();
});

it("does not replay earlier orders when a later agent fails", async () => {
	const earlier = vi.fn(() => [
		{
			participantId: "earlier",
			type: "LIMIT" as const,
			side: "BUY" as const,
			price: 99,
			quantity: 1,
		},
	]);
	const simulator = new Simulator({
		agents: [
			{ id: "earlier", step: earlier },
			agent({
				predict: async () => {
					throw new Error("offline");
				},
			}),
		],
	});
	await expect(simulator.runStepAsync()).rejects.toThrow("offline");
	expect(simulator.getOrderBookSnapshot().bids).toHaveLength(1);
	expect(simulator.getClock()).toBe(0);
	await expect(simulator.runStepAsync()).rejects.toThrow(
		"reset or recreate",
	);
	expect(earlier).toHaveBeenCalledTimes(1);
});
