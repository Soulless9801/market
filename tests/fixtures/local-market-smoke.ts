import assert from "node:assert/strict";
import { createServer } from "vite";
import { loadLocalModel } from "@/runtime/browser/localModel";
import { buildAgents, Simulator } from "@/simulation";
import { buildMarketViewModel } from "@/ui/ViewModel";

// Exercise the same Vite middleware and browser model adapter used by MarketView.
const server = await createServer({ server: { host: "127.0.0.1", port: 0 } });
const originalFetch = globalThis.fetch;
try {
	await server.listen();
	const address = server.httpServer!.address();
	assert(address && typeof address !== "string");
	const origin = `http://127.0.0.1:${address.port}`;
	globalThis.fetch = (input, init) => originalFetch(new URL(String(input), origin), init);
	const binding = await loadLocalModel();
	assert.equal(binding.alias, process.env.MARKET_MODEL_ALIAS);
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
}
