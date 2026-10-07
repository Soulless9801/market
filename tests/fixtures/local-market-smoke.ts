import assert from "node:assert/strict";
import { build, createServer, preview } from "vite";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { bindLocalModel, loadLocalModel, loadLocalModelCatalog } from "@/runtime/browser/localModel";
import { buildAgents, Simulator } from "@/simulation";
import { buildMarketViewModel } from "@/ui/ViewModel";

// Exercise the same Vite middleware and browser model adapter used by MarketView.
const isPreview = process.argv[2] === "preview";
const outDir = isPreview ? await mkdtemp(join(tmpdir(), "market-preview-")) : undefined;
if (isPreview) await build({ build: { outDir } });
const server = isPreview
	? await preview({ build: { outDir }, preview: { host: "127.0.0.1", port: 0 } })
	: await createServer({ server: { host: "127.0.0.1", port: 0 } });
const originalFetch = globalThis.fetch;
try {
	if ("listen" in server) await server.listen();
	const address = server.httpServer!.address();
	assert(address && typeof address !== "string");
	const origin = `http://127.0.0.1:${address.port}`;
	globalThis.fetch = (input, init) => originalFetch(new URL(String(input), origin), init);
	const page = await fetch("/");
	assert.equal(page.status, 200);
	if (isPreview) assert.match(await page.text(), /\/assets\/index-.*\.js/);
	const binding = await loadLocalModel();
	const catalog = await loadLocalModelCatalog();
	assert.equal(binding.alias, catalog.defaultAlias);
	if (process.env.MARKET_MODEL_ALIAS) assert.equal(binding.alias, process.env.MARKET_MODEL_ALIAS);
	// Each independently selected alias must work through the same browser adapter.
	for (const descriptor of catalog.models) {
		const selected = bindLocalModel(descriptor);
		assert.equal((await loadLocalModel(selected.alias)).featureLayout, selected.featureLayout);
		const run = new Simulator({ agents: buildAgents(0, 100, selected), referencePrice: 100 });
		for (let step = 0; step < 30; step++) await run.runStepAsync();
		assert.equal(run.getClock(), 30);
		assert(run.getParticpantPortfolios().some(p => p.participantId === "ml-10"));
	}
	const simulator = new Simulator({ agents: buildAgents(15, 100, binding), referencePrice: 100 });
	for (let step = 0; step < 30; step++) await simulator.runStepAsync();
	const view = buildMarketViewModel(
		simulator.getOrderBookSnapshot(), simulator.getStatistics(), simulator.getTradeHistory(),
		simulator.getParticpantPortfolios(), simulator.getClock(), simulator.getMidPriceHistory(),
	);
	const participant = view.participants.find(item => item.agentId === "ml-10");
	assert(participant);
	assert(participant.ordersSubmitted > 0);
	assert.equal(view.clock, 30);
	console.log("Local MarketView Python integration passed");
} finally {
	globalThis.fetch = originalFetch;
	await server.close();
	if (outDir) await rm(outDir, { recursive: true, force: true });
}
