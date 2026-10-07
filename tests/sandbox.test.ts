import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import {
	buildAgents, buildDefaultAgents, buildRandomAgents, DEFAULT_POPULATION_SEED,
	SANDBOX_ML_AGENT_ID, Simulator,
} from "@/simulation";
import { ModelSandbox } from "@/ui/ModelSandbox";

describe("sandbox population presets", () => {
	it("uses the default factory for reserved seed zero without changing other populations", () => {
		expect(DEFAULT_POPULATION_SEED).toBe(0);
		expect(buildAgents(DEFAULT_POPULATION_SEED)).toEqual(buildDefaultAgents(DEFAULT_POPULATION_SEED));
		for (const seed of [1, 15, 42]) expect(buildAgents(seed)).toEqual(buildRandomAgents(seed));
	});

	it("replays the default population with exactly one independently bound ML agent", async () => {
		const binding = { featureLayout: "mlp", model: { predict: async () => [0, 0, 1] } };
		const agents = buildAgents(DEFAULT_POPULATION_SEED, 100, binding);
		expect(agents.slice(0, -1)).toEqual(buildDefaultAgents(DEFAULT_POPULATION_SEED));
		expect(agents.filter(agent => agent.id === SANDBOX_ML_AGENT_ID)).toHaveLength(1);
		const first = new Simulator({ agents, referencePrice: 100 });
		const replay = new Simulator({ agents: buildAgents(DEFAULT_POPULATION_SEED, 100, binding), referencePrice: 100 });
		for (let step = 0; step < 30; step++) {
			await first.runStepAsync();
			await replay.runStepAsync();
		}
		expect(first.getTradeHistory()).toEqual(replay.getTradeHistory());
		expect(first.getParticpantPortfolios()).toEqual(replay.getParticpantPortfolios());
	});
});

it("renders arbitrary model aliases and only the supplied ML participant's stats", () => {
	const html = renderToStaticMarkup(createElement(ModelSandbox, {
		catalog: { defaultAlias: "third-release", models: ["first", "second", "third-release"].map(alias => ({ alias, metadata: {} })) },
		selectedAlias: "third-release", activeSeed: 0, isReady: true,
		participant: { agentId: SANDBOX_ML_AGENT_ID, pnl: 12.5, equity: 100012.5, cash: 90000, inventory: 100, marketValue: 10012.5, ordersSubmitted: 27 },
		onSelect: async () => {}
	}));
	expect(html).toContain('aria-pressed="true">third-release');
	expect(html).toContain("Default participants + ML agent");
	expect(html).toContain("$12.50");
	expect(html).toContain("Orders submitted</dt><dd>27");
});
