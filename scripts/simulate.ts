import { parseArgs } from "node:util";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createPythonSimulation } from "@/runtime/python/createPythonSimulation";

export async function main(args = process.argv.slice(2)): Promise<void> {
	// This is the Node composition entry point. It owns process lifetime; the
	// browser-facing simulator and agent modules contain no Python spawn logic.
	const { values } = parseArgs({
		args,
		options: {
			checkpoint: { type: "string" },
			python: { type: "string", default: ".venv/bin/python" },
			model: { type: "string", default: "market-side-v1" },
			steps: { type: "string", default: "100" },
			seed: { type: "string", default: "15" },
		},
	});
	const steps = Number(values.steps),
		seed = Number(values.seed);
	if (
		!values.checkpoint ||
		!Number.isSafeInteger(steps) ||
		steps < 1 ||
		!Number.isSafeInteger(seed) ||
		seed < 0 ||
		seed > 0xffffffff
	) {
		throw new Error(
			"Usage: npm run simulate -- --checkpoint PATH [--python PATH] [--steps N] [--seed N] [--model ALIAS]",
		);
	}
	const session = await createPythonSimulation({
		seed,
		modelAlias: values.model,
		python: {
			pythonExecutable: values.python,
			checkpoints: { [values.model]: values.checkpoint },
		},
	});
	try {
		for (let step = 0; step < steps; step++)
			await session.simulator.runStepAsync();
		console.log(
			JSON.stringify({
				clock: session.simulator.getClock(),
				statistics: session.simulator.getStatistics(),
				participants:
					session.simulator.getParticpantPortfolios(),
			}),
		);
	} finally {
		await session.shutdown();
	}
}

if (
	process.argv[1] &&
	import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
	main().catch((error) => {
		console.error(error);
		process.exitCode = 1;
	});
}
