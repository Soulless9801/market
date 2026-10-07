import { DEFAULT_POPULATION_SEED } from "@/simulation";
import type { LocalModelCatalog } from "@/runtime/local-inference";
import type { ParticipantSnapshot } from "./ViewModel";
import { formatPrice, formatQuantity } from "./format";

interface ModelSandboxProps {
	catalog: LocalModelCatalog | null;
	selectedAlias: string | null;
	activeSeed: number;
	participant?: ParticipantSnapshot;
	isReady: boolean;
	onSelect: (alias: string) => Promise<void>;
}

/** Model choices are deployment aliases supplied by the host, not architecture names. */
export function ModelSandbox({ catalog, selectedAlias, activeSeed, participant, isReady, onSelect }: ModelSandboxProps) {
	return <section className="model-sandbox" aria-label="ML sandbox">
		<div className="model-sandbox-heading">
			<div>
				<h2 className="market-panel-heading">ML Agent</h2>
				<p>choose your agent</p>
			</div>
		</div>
		<div className="model-choices" role="group" aria-label="Sandbox model">
			{catalog?.models.map(model => <button
				key={model.alias} className="market-control model-choice"
				aria-pressed={selectedAlias === model.alias}
				onClick={() => { void onSelect(model.alias); }}
			>{model.alias}</button>)}
			{!catalog && <span className="sandbox-run-description">Model catalog unavailable until inference connects.</span>}
		</div>
		<p className="sandbox-run-description">
			{isReady ? `${selectedAlias} · Seed ${activeSeed} · ${activeSeed === DEFAULT_POPULATION_SEED ? "Default participants" : "Random participants"} + ML agent` : "Waiting for a model session…"}
		</p>
		<dl className="ml-agent-stats" aria-label="ML agent statistics">
			{[
				["PnL", participant ? formatPrice(participant.pnl) : "—"],
				["Equity", participant ? formatPrice(participant.equity) : "—"],
				["Cash", participant ? formatPrice(participant.cash) : "—"],
				["Inventory", participant ? formatQuantity(participant.inventory) : "—"],
				["Market value", participant ? formatPrice(participant.marketValue) : "—"],
				["Orders submitted", participant ? formatQuantity(participant.ordersSubmitted) : "—"],
			].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
		</dl>
	</section>;
}
