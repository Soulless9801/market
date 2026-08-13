import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, ReactElement } from "react";

import type { OrderBookSnapshot } from "../engine";
import { buildDefaultAgents, Simulator } from "../simulation";
import { buildMarketViewModel, type MarketViewModel } from "./marketViewModel";
import { calculateMidPrice } from "../engine";

const DEFAULT_SEED = 7;
const DEFAULT_REFERENCE_PRICE = 100;

function createSimulator(seed: number): Simulator {
	return new Simulator({
		agents: buildDefaultAgents(seed, DEFAULT_REFERENCE_PRICE),
		referencePrice: DEFAULT_REFERENCE_PRICE,
	});
}

function getMidPrice(snapshot: OrderBookSnapshot): number {
	return calculateMidPrice(snapshot, DEFAULT_REFERENCE_PRICE);
}

function useSimulationController() {
	const simulatorRef = useRef<Simulator | null>(null);
	const [seed, setSeed] = useState(DEFAULT_SEED);
	const [isRunning, setIsRunning] = useState(true);
	const [playbackSpeed, setPlaybackSpeed] = useState(1);
	const [viewModel, setViewModel] = useState<MarketViewModel>(() => {
		const simulator = createSimulator(DEFAULT_SEED);
		simulatorRef.current = simulator;
		return buildMarketViewModel(
			simulator.getOrderBookSnapshot(),
			simulator.getTradeHistory(),
			simulator.getParticpantPortfolios(),
			simulator.getClock(),
			[DEFAULT_REFERENCE_PRICE],
		);
	});

	const resetSimulation = useCallback(() => {
		const simulator = createSimulator(seed);
		simulatorRef.current = simulator;
		const initialSnapshot = simulator.getOrderBookSnapshot();
		setViewModel(
			buildMarketViewModel(
				initialSnapshot,
				simulator.getTradeHistory(),
				simulator.getParticpantPortfolios(),
				simulator.getClock(),
				[getMidPrice(initialSnapshot)],
			),
		);
	}, [seed]);

	useEffect(() => {
		resetSimulation();
	}, [resetSimulation]);

	useEffect(() => {
		if (!isRunning) {
			return undefined;
		}

		const intervalId = window.setInterval(() => {
			const simulator = simulatorRef.current;
			if (!simulator) {
				return;
			}

			simulator.runStep();
			const snapshot = simulator.getOrderBookSnapshot();
			const nextMidPrice = getMidPrice(snapshot);
			setViewModel((previous) =>
				buildMarketViewModel(
					snapshot,
					simulator.getTradeHistory(),
					simulator.getParticpantPortfolios(),
					simulator.getClock(),
					[...previous.midPriceSeries.slice(-39), nextMidPrice],
				),
			);
		}, 1000 / playbackSpeed);

		return () => window.clearInterval(intervalId);
	}, [isRunning, playbackSpeed]);

	return {
		viewModel,
		isRunning,
		seed,
		playbackSpeed,
		setSeed,
		setIsRunning,
		setPlaybackSpeed,
		resetSimulation,
	};
}

function formatPrice(value: number | null): string {
	if (value === null) {
		return "—";
	}
	return value.toFixed(2);
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
		resetSimulation,
	} = useSimulationController();

	const maxHistory = useMemo(() => Math.max(...viewModel.midPriceSeries, DEFAULT_REFERENCE_PRICE), [viewModel.midPriceSeries]);
	const minHistory = useMemo(() => Math.min(...viewModel.midPriceSeries, DEFAULT_REFERENCE_PRICE), [viewModel.midPriceSeries]);

	return (
		<div style={{ minHeight: "100vh", background: "#060b16", color: "#f2f5ff", padding: "24px", fontFamily: "Inter, system-ui, sans-serif" }}>
			<div style={{ maxWidth: "1400px", margin: "0 auto", display: "grid", gap: "16px" }}>
				<header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingBottom: "8px", borderBottom: "1px solid #23304d" }}>
					<div>
						<h1 style={{ margin: 0, fontSize: "28px", letterSpacing: "0.03em" }}>Synthetic Market Monitor</h1>
						<p style={{ margin: "6px 0 0", color: "#86a0c9" }}>Live order flow, liquidity, and trade activity</p>
					</div>
					<div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
						<button onClick={() => setIsRunning((value) => !value)} style={buttonStyle}>
							{isRunning ? "Pause" : "Start"}
						</button>
						<button onClick={resetSimulation} style={buttonStyle}>Reset</button>
						<div style={{ fontSize: "13px", color: "#86a0c9" }}>Composition: 1 MM · 2 retail</div>
						<label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px", color: "#86a0c9" }}>
							Seed
							<input
								type="number"
								value={seed}
								onChange={(event) => setSeed(Number(event.target.value))}
								style={{ ...inputStyle, width: "70px" }}
							/>
						</label>
						<select value={playbackSpeed} onChange={(event) => setPlaybackSpeed(Number(event.target.value))} style={inputStyle}>
							<option value={0.5}>0.5x</option>
							<option value={1}>1x</option>
							<option value={2}>2x</option>
							<option value={5}>5x</option>
							<option value={10}>10x</option>
						</select>
					</div>
				</header>

				<section style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: "16px" }}>
					<div style={panelStyle}>
						<div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
							<div>
								<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "#86a0c9", textTransform: "uppercase" }}>Order book</div>
								<div style={{ fontSize: "14px", color: "#f2f5ff" }}>Simulation time {viewModel.clock}</div>
							</div>
							<div style={{ textAlign: "right" }}>
								<div style={{ fontSize: "12px", color: "#86a0c9" }}>Best bid</div>
								<div style={{ fontSize: "18px", fontWeight: 600 }}>{formatPrice(viewModel.bids[0]?.price ?? null)}</div>
								<div style={{ fontSize: "12px", color: "#86a0c9" }}>Best ask</div>
								<div style={{ fontSize: "18px", fontWeight: 600 }}>{formatPrice(viewModel.asks[0]?.price ?? null)}</div>
							</div>
						</div>
						<div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
							<div>
								<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "#ff6b6b", textTransform: "uppercase", marginBottom: "8px" }}>ASK</div>
								{viewModel.asks.map((row) => (
									<div key={`${row.side}-${row.price}`} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
										<div style={{ width: "70px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{formatPrice(row.price)}</div>
										<div style={{ flex: 1, height: "10px", background: "rgba(255,107,107,0.16)", borderRadius: "999px", overflow: "hidden" }}>
											<div style={{ width: `${row.barWidth}%`, height: "100%", background: "#ff6b6b", borderRadius: "999px" }} />
										</div>
										<div style={{ width: "70px", fontVariantNumeric: "tabular-nums", color: "#86a0c9" }}>{formatQuantity(row.quantity)}</div>
									</div>
								))}
							</div>
							<div>
								<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "#4fd1c5", textTransform: "uppercase", marginBottom: "8px" }}>BID</div>
								{viewModel.bids.map((row) => (
									<div key={`${row.side}-${row.price}`} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
										<div style={{ width: "70px", fontVariantNumeric: "tabular-nums" }}>{formatPrice(row.price)}</div>
										<div style={{ flex: 1, height: "10px", background: "rgba(79,209,197,0.16)", borderRadius: "999px", overflow: "hidden" }}>
											<div style={{ width: `${row.barWidth}%`, height: "100%", background: "#4fd1c5", borderRadius: "999px" }} />
										</div>
										<div style={{ width: "70px", fontVariantNumeric: "tabular-nums", color: "#86a0c9" }}>{formatQuantity(row.quantity)}</div>
									</div>
								))}
							</div>
						</div>
					</div>

					<div style={{ display: "grid", gap: "16px" }}>
						<div style={panelStyle}>
							<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "#86a0c9", textTransform: "uppercase", marginBottom: "12px" }}>Market statistics</div>
							<div style={{ display: "grid", gap: "10px" }}>
								<Metric label="Midprice" value={formatPrice(viewModel.midPrice)} />
								<Metric label="Spread" value={viewModel.spread === null ? "—" : formatPrice(viewModel.spread)} />
								<Metric label="Volume" value={formatQuantity(viewModel.totalVolume)} />
								<Metric label="Trades" value={viewModel.tradeCount.toString()} />
							</div>
						</div>

						<div style={panelStyle}>
							<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "#86a0c9", textTransform: "uppercase", marginBottom: "12px" }}>Order imbalance</div>
							<div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
								<div style={{ flex: 1, height: "10px", background: "rgba(255,255,255,0.08)", borderRadius: "999px", overflow: "hidden" }}>
									<div style={{ width: `${viewModel.imbalance.bidPercent}%`, height: "100%", background: "#4fd1c5", borderRadius: "999px" }} />
								</div>
								<div style={{ minWidth: "88px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{viewModel.imbalance.bidPercent.toFixed(1)}% bid</div>
							</div>
							<div style={{ color: "#86a0c9", fontSize: "14px" }}>Bid liquidity: {viewModel.imbalance.bidVolume.toLocaleString()} | Ask liquidity: {viewModel.imbalance.askVolume.toLocaleString()}</div>
						</div>
					</div>
				</section>

				<section style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: "16px" }}>
					<div style={panelStyle}>
						<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "#86a0c9", textTransform: "uppercase", marginBottom: "12px" }}>Midprice</div>
						<svg viewBox="0 0 420 220" style={{ width: "100%", height: "280px" }}>
							<rect x="0" y="0" width="420" height="220" fill="rgba(8,17,32,0.55)" rx="10" />
							{renderChartGrid(420, 220, viewModel.midPriceSeries, viewModel.clock, maxHistory, minHistory)}
							<path
								d={buildPath(viewModel.midPriceSeries, { width: 380, height: 160, maxValue: maxHistory, minValue: minHistory })}
								fill="none"
								stroke="#4fd1c5"
								strokeWidth="2.5"
							/>
							<text x="210" y="208" textAnchor="middle" fill="#86a0c9" fontSize="11">Simulation step</text>
							<text x="10" y="112" textAnchor="middle" fill="#86a0c9" fontSize="11" transform="rotate(-90 10 112)">Midprice</text>
						</svg>
					</div>

					<div style={panelStyle}>
						<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "#86a0c9", textTransform: "uppercase", marginBottom: "12px" }}>Participants</div>
						<div style={{ display: "grid", gap: "8px" }}>
							{viewModel.participants.map((participant) => (
								<div key={participant.agentId} style={{ padding: "10px 12px", border: "1px solid #23304d", borderRadius: "8px", background: "#081120" }}>
									<div style={{ fontWeight: 600 }}>{participant.agentId}</div>
									<div style={{ marginTop: "4px", color: "#86a0c9", fontSize: "14px" }}>PNL: {participant.pnl}</div>
									<div style={{ color: "#86a0c9", fontSize: "14px" }}>Inventory: {participant.inventory}</div>
									<div style={{ color: "#86a0c9", fontSize: "14px" }}>Orders Submitted: {participant.ordersSubmitted}</div>
									<div style={{ color: "#86a0c9", fontSize: "14px" }}>Cash: {participant.cash}</div>
								</div>
							))}
						</div>
					</div>
				</section>

				<section style={panelStyle}>
					<div style={{ fontSize: "12px", letterSpacing: "0.16em", color: "#86a0c9", textTransform: "uppercase", marginBottom: "12px" }}>Trade tape</div>
					<div style={{ display: "grid", gap: "8px" }}>
						{viewModel.trades.map((trade) => (
							<div key={trade.id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", background: "#081120", borderRadius: "8px", fontFamily: "ui-monospace, SFMono-Regular, monospace" }}>
								<div>{trade.timestamp} {trade.side} {trade.quantity} @ {formatPrice(trade.price)}</div>
								<div style={{ color: "#86a0c9" }}>{trade.side}</div>
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
		<div style={{ display: "flex", justifyContent: "space-between", padding: "8px 10px", border: "1px solid #23304d", borderRadius: "8px", background: "#081120" }}>
			<span style={{ color: "#86a0c9" }}>{label}</span>
			<span style={{ fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>{value}</span>
		</div>
	);
}

function buildPath(series: number[], dimensions: { width: number; height: number; maxValue: number; minValue: number }) {
	if (series.length === 0) {
		return "";
	}

	const leftPadding = 20;
	const topPadding = 20;
	const innerWidth = dimensions.width;
	const innerHeight = dimensions.height;
	const points = series.map((value, index) => {
		const x = leftPadding + (index / Math.max(1, series.length - 1)) * innerWidth;
		const normalized = (value - dimensions.minValue) / Math.max(1, dimensions.maxValue - dimensions.minValue);
		const y = topPadding + innerHeight - normalized * innerHeight;
		return `${x},${y}`;
	});

	return `M ${points.join(" L ")}`;
}

function renderChartGrid(width: number, height: number, series: number[], currentStep: number, maxValue: number, minValue: number) {
	const ticks = 5;
	const xTickCount = Math.min(6, Math.max(2, series.length));
	const innerWidth = width - 40;
	const innerHeight = height - 40;
	const lines: ReactElement[] = [];

	for (let index = 0; index < ticks; index += 1) {
		const yRatio = index / (ticks - 1);
		const x = 20;
		const y = 20 + yRatio * innerHeight;
		const value = minValue + (maxValue - minValue) * (1 - yRatio);
		lines.push(
			<g key={`y-${index}`}>
				<line x1={x} x2={width - 20} y1={y} y2={y} stroke="#23304d" strokeWidth="1" />
				<text x="8" y={y + 4} fill="#86a0c9" fontSize="10">{value.toFixed(1)}</text>
			</g>,
		);
	}

	for (let index = 0; index < xTickCount; index += 1) {
		const xRatio = index / Math.max(1, xTickCount - 1);
		const sampleIndex = Math.round(xRatio * Math.max(0, series.length - 1));
		const x = 20 + xRatio * innerWidth;
		const y = height - 20;
		const step = currentStep - Math.max(0, series.length - 1 - sampleIndex);
		lines.push(
			<g key={`x-${index}`}>
				<line x1={x} x2={x} y1="20" y2={y} stroke="#23304d" strokeWidth="1" />
				<text x={x} y={y + 16} textAnchor="middle" fill="#86a0c9" fontSize="10">{step}</text>
			</g>,
		);
	}

	return <>{lines}</>;
}

const panelStyle: CSSProperties = {
	background: "#0d162b",
	padding: "16px",
	borderRadius: "12px",
	border: "1px solid #23304d",
};

const buttonStyle: CSSProperties = {
	padding: "8px 12px",
	borderRadius: "8px",
	border: "1px solid #23304d",
	background: "#0d162b",
	color: "#f2f5ff",
	cursor: "pointer",
};

const inputStyle: CSSProperties = {
	padding: "8px 10px",
	borderRadius: "8px",
	border: "1px solid #23304d",
	background: "#081120",
	color: "#f2f5ff",
};

export default MarketMonitor;
