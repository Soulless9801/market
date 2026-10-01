import type { CSSProperties } from "react";
import { useSimulationController } from "./useSimulationController";
import "./MarketMonitor.css";
import { CandlestickChart } from "./CandlestickChart";

function formatPrice(value: number | null): string {
	if (value === null) {
		return "—";
	}
	return "$" + value.toFixed(2);
}

function formatCash(value: number): string {
	if (value === null) {
		return "—";
	}
	if (value < 0) {
		return "-$" + Math.abs(value).toFixed(2);
	}
	return "$" + value.toFixed(2);
}

function formatQuantity(value: number): string {
	return value.toLocaleString();
}

function MarketMonitor() {
	const {
		viewModel,
		isRunning,
		seed,
		playbackSpeed,
		setSeed,
		setIsRunning,
		setPlaybackSpeed,
		reset,
		modelStatus,
		error,
		isReady,
	} = useSimulationController();

	return (
		<div className="market-monitor" style={{ minHeight: "100vh", background: "var(--market-background)", color: "var(--market-text)", padding: "24px", fontFamily: "Inter, system-ui, sans-serif" }}>
			<div style={{ maxWidth: "1400px", margin: "0 auto", display: "grid", gap: "16px" }}>
				<header className="market-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingBottom: "20px", borderBottom: "1px solid var(--market-border)" }}>
					<div>
						<h1 style={{ margin: 0, fontSize: "24px", fontWeight: 500, letterSpacing: "-0.02em" }}>Synthetic Market Monitor</h1>
						<p style={{ margin: "6px 0 0", color: "var(--market-muted)" }}>negative cash and inventory is (totally) a feature</p>
					</div>
					<div className="market-controls" style={{ display: "flex", gap: "8px", alignItems: "center" }}>
						<button className="market-control" disabled={!isReady} onClick={() => setIsRunning((value) => !value)} style={buttonStyle}>
							{isRunning ? "Pause" : "Start"}
						</button>
						<button className="market-control" onClick={() => { void reset(); }} style={buttonStyle}>Reset</button>
						<label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px", color: "var(--market-muted)" }}>
							Seed
							<input
								className="market-control"
								type="number"
								value={seed}
								onChange={(event) => setSeed(Number(event.target.value))}
								style={{ ...inputStyle, width: "70px" }}
							/>
						</label>
						<select className="market-control" aria-label="Playback speed" value={playbackSpeed} onChange={(event) => setPlaybackSpeed(Number(event.target.value))} style={inputStyle}>
							<option value={0.5}>0.5x</option>
							<option value={1}>1x</option>
							<option value={2}>2x</option>
							<option value={5}>5x</option>
							<option value={10}>10x</option>
							<option value={100}>100x</option>
						</select>
					</div>
				</header>

				<div className="market-status" role={error ? "alert" : "status"}>
					{error ? `Local ML Unavailable` : modelStatus}
				</div>

				<section className="market-grid">
					<div style={panelStyle}>
						<div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
							<div>
								<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "var(--market-muted)", textTransform: "uppercase" }}>Order book</div>
								<div style={{ fontSize: "14px", color: "var(--market-text)" }}>Simulation Time {viewModel.clock}</div>
							</div>
							<div style={{ textAlign: "right" }}>
								<div style={{ fontSize: "12px", color: "var(--market-muted)" }}>Best Bid</div>
								<div style={{ fontSize: "18px", fontWeight: 600 }}>{formatPrice(viewModel.bids[0]?.price ?? null)}</div>
								<div style={{ fontSize: "12px", color: "var(--market-muted)" }}>Best Ask</div>
								<div style={{ fontSize: "18px", fontWeight: 600 }}>{formatPrice(viewModel.asks[0]?.price ?? null)}</div>
							</div>
						</div>
						<div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
							<div>
								<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "var(--market-ask)", textTransform: "uppercase", marginBottom: "8px" }}>ASK</div>
								{viewModel.asks.map((row) => (
									<div key={`${row.side}-${row.price}`} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
										<div style={{ width: "70px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{formatPrice(row.price)}</div>
										<div style={{ flex: 1, height: "10px", background: "var(--market-track)", borderRadius: 0, overflow: "hidden" }}>
											<div style={{ width: `${row.barWidth}%`, height: "100%", background: "var(--market-ask)", borderRadius: 0 }} />
										</div>
										<div style={{ width: "70px", fontVariantNumeric: "tabular-nums", color: "var(--market-muted)" }}>{formatQuantity(row.quantity)}</div>
									</div>
								))}
							</div>
							<div>
								<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "var(--market-bid)", textTransform: "uppercase", marginBottom: "8px" }}>BID</div>
								{viewModel.bids.map((row) => (
									<div key={`${row.side}-${row.price}`} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
										<div style={{ width: "70px", fontVariantNumeric: "tabular-nums" }}>{formatPrice(row.price)}</div>
										<div style={{ flex: 1, height: "10px", background: "var(--market-track)", borderRadius: 0, overflow: "hidden" }}>
											<div style={{ width: `${row.barWidth}%`, height: "100%", background: "var(--market-bid)", borderRadius: 0 }} />
										</div>
										<div style={{ width: "70px", fontVariantNumeric: "tabular-nums", color: "var(--market-muted)" }}>{formatQuantity(row.quantity)}</div>
									</div>
								))}
							</div>
						</div>
					</div>

					<div style={{ display: "grid", gap: "16px" }}>
						<div style={panelStyle}>
							<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "var(--market-muted)", textTransform: "uppercase", marginBottom: "12px" }}>Market statistics</div>
							<div style={{ display: "grid", gap: "10px" }}>
								<Metric label="Midprice" value={formatPrice(viewModel.midPrice)} />
								<Metric label="Spread" value={viewModel.spread === null ? "—" : formatPrice(viewModel.spread)} />
								<Metric label="Volume" value={formatQuantity(viewModel.totalVolume)} />
								<Metric label="Trades" value={viewModel.tradeCount.toString()} />
							</div>
						</div>

						<div style={panelStyle}>
							<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "var(--market-muted)", textTransform: "uppercase", marginBottom: "12px" }}>Order imbalance</div>
							<div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
								<div style={{ flex: 1, height: "10px", background: "var(--market-track)", borderRadius: 0, overflow: "hidden" }}>
									<div style={{ width: `${viewModel.imbalance.bidPercent}%`, height: "100%", background: "var(--market-bid)", borderRadius: 0 }} />
								</div>
								<div style={{ minWidth: "88px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{viewModel.imbalance.bidPercent.toFixed(1)}% bid</div>
							</div>
							<div style={{ color: "var(--market-muted)", fontSize: "14px" }}>Bid liquidity: {viewModel.imbalance.bidVolume.toLocaleString()} | Ask liquidity: {viewModel.imbalance.askVolume.toLocaleString()}</div>
						</div>
					</div>
				</section>

				<section className="market-grid">
					<div style={panelStyle}>
						<CandlestickChart prices={viewModel.midPriceSeries} currentStep={viewModel.clock} />
					</div>

					<div style={panelStyle}>
						<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "var(--market-muted)", textTransform: "uppercase", marginBottom: "12px" }}>Participants</div>
						<div style={{ display: "grid", gap: "8px", scrollBehavior: "smooth", maxHeight: "280px", overflowY: "auto" }}>
							{viewModel.participants.map((participant) => (
								<div key={participant.agentId} style={{ padding: "10px 12px", border: "1px solid var(--market-border)", borderRadius: 0, background: "var(--market-surface)" }}>
									<div style={{ fontWeight: 600 }}>{participant.agentId}</div>
									<div style={{ marginTop: "4px", color: "var(--market-muted)", fontSize: "14px" }}>PNL: {formatCash(participant.pnl)}</div>
									<div style={{ color: "var(--market-muted)", fontSize: "14px" }}>Inventory: {participant.inventory}</div>
									<div style={{ color: "var(--market-muted)", fontSize: "14px" }}>Orders Submitted: {participant.ordersSubmitted}</div>
									<div style={{ color: "var(--market-muted)", fontSize: "14px" }}>Cash: {formatCash(participant.cash)}</div>
								</div>
							))}
						</div>
					</div>
				</section>

				<section style={panelStyle}>
					<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "var(--market-muted)", textTransform: "uppercase", marginBottom: "12px" }}>Trade tape</div>
					<div style={{ display: "grid", gap: "8px" }}>
						{viewModel.trades.map((trade) => (
							<div key={trade.id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", background: "var(--market-surface)", borderRadius: 0, fontFamily: "ui-monospace, SFMono-Regular, monospace" }}>
								<div>{trade.timestamp} {trade.side} {trade.quantity} @ {formatPrice(trade.price)}</div>
								<div style={{ color: "var(--market-muted)" }}>{trade.side}</div>
							</div>
						))}
					</div>
				</section>
			</div>
		</div>
	);
}

function Metric({ label, value }: { label: string; value: string }) {
	return (
		<div style={{ display: "flex", justifyContent: "space-between", padding: "8px 10px", border: "1px solid var(--market-border)", borderRadius: 0, background: "var(--market-surface)" }}>
			<span style={{ color: "var(--market-muted)" }}>{label}</span>
			<span style={{ fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>{value}</span>
		</div>
	);
}

const panelStyle: CSSProperties = {
	background: "var(--market-panel)",
	padding: "16px",
	borderRadius: 0,
	border: "1px solid var(--market-border)",
};

const buttonStyle: CSSProperties = {
	padding: "8px 12px",
	borderRadius: 0,
	border: "1px solid var(--market-border)",
	background: "var(--market-panel)",
	color: "var(--market-text)",
};

const inputStyle: CSSProperties = {
	appearance: "none",
	padding: "8px 12px",
	borderRadius: 0,
	border: "1px solid var(--market-border)",
	background: "var(--market-surface)",
	color: "var(--market-text)",
};

export default MarketMonitor;
