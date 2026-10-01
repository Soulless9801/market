import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { setTimeout as delay } from "node:timers/promises";
import { PythonInferenceClient } from "@/runtime/python/PythonInferenceClient";
import { validateSideModelMetadata } from "@/simulation/ml/model-binding";
import {
	FeatureManager,
	MarketMakerAgent,
	MLTraderAgent,
	Simulator,
} from "@/simulation";

const [pythonExecutable, checkpoint, fixturePath, pidFile] =
	process.argv.slice(2);
const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as {
	input: number[];
	expected: number[];
};
const client = new PythonInferenceClient({
	pythonExecutable,
	checkpoints: {
		first: checkpoint,
		second: checkpoint,
		independent: checkpoint,
	},
});
let predictionsChecked = 0;
let mlTrades = 0;
try {
	await client.start();
	const pid = Number(readFileSync(pidFile, "utf8"));
	assert.ok(Number.isSafeInteger(pid) && pid > 0 && pid !== process.pid);
	for (const alias of ["first", "second", "independent"]) {
		assert.deepEqual(
			await client.predict(alias, fixture.input),
			fixture.expected,
		);
	}
	const metadata = await client.getMetadata("independent");
	assert.deepEqual(Object.keys(metadata as object).sort(), [
		"classNames",
		"features",
		"inputShape",
		"outputType",
	]);
	const featureLayout = validateSideModelMetadata(metadata);
	const definition = FeatureManager.describe(featureLayout);
	const builder = FeatureManager.create(featureLayout);
	const bound = client.model("independent");
	const replays = [];
	for (let replay = 0; replay < 2; replay++) {
		let expectedInput: number[] = [];
		const ml = new MLTraderAgent("ml-audit", {
			featureLayout,
			referencePrice: 100,
			spread: 2,
			quantity: 5,
			seed: 42,
			model: {
				predict(input) {
					// This was computed before the step, before any agent submits an order.
					assert.deepEqual(input, expectedInput);
					assert.ok(
						input.every(
							(value) =>
								typeof value ===
									"number" &&
								Number.isFinite(
									value,
								),
						),
					);
					predictionsChecked++;
					return bound.predict(input);
				},
			},
		});
		const simulator = new Simulator({
			agents: [
				new MarketMakerAgent("liquidity", {
					referencePrice: 100,
					spread: 2,
					quantity: 20,
				}),
				ml,
			],
		});
		for (
			let index = 0;
			index < definition.minimumWarmupSteps + 60;
			index++
		) {
			const observation = simulator.getObservableContext(
				definition.tradeHistoryLimit,
				definition.priceHistoryLimit,
				definition.bookDepth,
			);
			assert.ok(!("portfolio" in observation));
			if (index >= definition.minimumWarmupSteps)
				expectedInput = builder.build(observation);
			await simulator.runStepAsync();
			assert.equal(simulator.getClock(), index + 1);
		}
		const trades = simulator
			.getTradeHistory()
			.filter(
				(trade) =>
					trade.buyerParticipantId === ml.id ||
					trade.sellerParticipantId === ml.id,
			);
		assert.ok(
			trades.length > 0,
			"Trained model must execute a real trade in this fixed regression scenario",
		);
		mlTrades += trades.length;
		replays.push({
			trades,
			events: simulator.getEvents(),
			portfolios: simulator.getParticpantPortfolios(),
			history: simulator.getMidPriceHistory(),
		});
	}
	assert.deepEqual(replays[0], replays[1]);
	assert.equal(
		Number(readFileSync(pidFile, "utf8")),
		pid,
		"Inference process was restarted",
	);

	const failing = new Simulator({
		agents: [
			new MLTraderAgent("ml-crash", {
				featureLayout,
				model: bound,
				referencePrice: 100,
				spread: 2,
				quantity: 5,
				seed: 42,
			}),
		],
	});
	for (let step = 0; step < definition.minimumWarmupSteps; step++)
		await failing.runStepAsync();
	process.kill(pid, "SIGSTOP");
	const before = {
		clock: failing.getClock(),
		book: failing.getOrderBookSnapshot(),
		history: failing.getMidPriceHistory(),
		events: failing.getEvents(),
	};
	const pending = failing.runStepAsync();
	const rejected = assert.rejects(pending, /Python/i);
	await delay(30); // Python is stopped: the pending request cannot complete.
	assert.equal(failing.getClock(), before.clock);
	assert.deepEqual(failing.getOrderBookSnapshot(), before.book);
	assert.deepEqual(failing.getMidPriceHistory(), before.history);
	assert.deepEqual(failing.getEvents(), before.events);
	process.kill(pid, "SIGKILL");
	await rejected;
	assert.equal(failing.getClock(), before.clock);
	assert.deepEqual(failing.getOrderBookSnapshot(), before.book);
	await assert.rejects(failing.runStepAsync(), /reset or recreate/);
	console.log(
		JSON.stringify({
			predictionsChecked,
			mlTrades,
			sameProcess: true,
			runtimeReplayEqual: true,
			pythonCrashRejected: true,
		}),
	);
} finally {
	await client.shutdown();
}
