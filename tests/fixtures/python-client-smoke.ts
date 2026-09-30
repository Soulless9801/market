import { createPythonSimulation } from "@/runtime/python/createPythonSimulation";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { PythonInferenceClient } from "@/runtime/python/PythonInferenceClient";
import type { PythonInferenceOptions } from "@/runtime/python/PythonInferenceClient";

const [pythonExecutable, checkpoint, fixturePath] = process.argv.slice(2);
const { input, expected } = JSON.parse(readFileSync(fixturePath, "utf8")) as {
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
try {
	await Promise.all([client.start(), client.start()]);
	for (const alias of ["first", "second", "independent"]) {
		assert.deepEqual(
			await client.model(alias).predict(input),
			expected,
		);
	}
	assert.deepEqual(
		await Promise.all([
			client.predict("first", input),
			client.predict("second", input),
		]),
		[expected, expected],
	);
	await assert.rejects(client.predict("absent", input), /Unknown model/);
	await assert.rejects(
		client.predict("first", [1]),
		/Expected.*input values/,
	);
	assert.deepEqual(await client.predict("first", input), expected);
} finally {
	await client.shutdown();
}
await assert.rejects(client.start(), /shut down/);

const brokenOptions: PythonInferenceOptions[] = [
	{ pythonExecutable, checkpoints: { absent: checkpoint + ".missing" } },
	{
		pythonExecutable: checkpoint + ".not-an-executable",
		checkpoints: { first: checkpoint },
	},
];
for (const options of brokenOptions) {
	const broken = new PythonInferenceClient(options);
	try {
		await assert.rejects(broken.start(), /Python/);
	} finally {
		await broken.shutdown();
	}
}
console.log(
	"Python client integration passed: parity, aliases, concurrency, recovery, startup, shutdown",
);

const simulations = [];
for (let run = 0; run < 2; run++) {
	const session = await createPythonSimulation({
		python: {
			pythonExecutable,
			checkpoints: { independent: checkpoint },
		},
		modelAlias: "independent",
		seed: 7,
	});
	try {
		for (let step = 0; step < 30; step++)
			await session.simulator.runStepAsync();
		assert.equal(session.simulator.getClock(), 30);
		assert.ok(
			session.simulator
				.getEvents()
				.some(
					(event) =>
						event.type === "agent-step" &&
						event.agentId === "ml-10",
				),
		);
		simulations.push({
			events: session.simulator.getEvents(),
			participants:
				session.simulator.getParticpantPortfolios(),
			history: session.simulator.getMidPriceHistory(),
		});
	} finally {
		await session.shutdown();
	}
}
assert.deepEqual(simulations[0], simulations[1]);
console.log(
	"Python simulation integration passed: sequential steps, deterministic results, cleanup",
);
