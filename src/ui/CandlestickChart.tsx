import { buildMidPriceCandles, CANDLE_STEPS, MAX_VISIBLE_CANDLES } from "./candles";

const WIDTH = 600;
const HEIGHT = 250;
const LEFT = 66;
const RIGHT = 18;
const TOP = 18;
const BOTTOM = 44;

export function CandlestickChart({ prices, currentStep }: { prices: readonly number[]; currentStep: number }) {
	const candles = buildMidPriceCandles(prices, currentStep);
	const last = candles.at(-1);
	const low = candles.length ? Math.min(...candles.map(candle => candle.low)) : 100;
	const high = candles.length ? Math.max(...candles.map(candle => candle.high)) : 100;
	// A shared padded scale keeps flat candles centered and wicks away from the plot edges.
	const padding = Math.max((high - low) * 0.12, 0.01);
	const minimum = low - padding;
	const maximum = high + padding;
	const plotHeight = HEIGHT - TOP - BOTTOM;
	const plotWidth = WIDTH - LEFT - RIGHT;
	const slotWidth = plotWidth / Math.max(12, candles.length);
	const bodyWidth = Math.min(16, slotWidth * 0.6);
	const y = (price: number) => TOP + (maximum - price) / (maximum - minimum) * plotHeight;
	const x = (index: number) => LEFT + (index + 0.5) * slotWidth;
	const labelEvery = Math.max(1, Math.ceil(candles.length / 6));

	return (
		<div>
			<div className="market-chart-heading">
				<span>Midprice candles</span>
				<span className="market-chart-legend">{CANDLE_STEPS} steps / candle · hollow up · filled down</span>
			</div>
			<svg role="img" aria-label="Midprice candlestick chart" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} style={{ width: "100%", height: "250px" }}>
				<title>Midprice candlesticks</title>
				<desc>Open, high, low and close of sampled midprices over {CANDLE_STEPS} simulation steps. Latest {MAX_VISIBLE_CANDLES} candles; the last candle may still be forming.</desc>
				<rect width={WIDTH} height={HEIGHT} fill="var(--market-surface)" />
				{Array.from({ length: 5 }, (_, index) => {
					const price = maximum - index / 4 * (maximum - minimum);
					return <g key={index}>
						<line x1={LEFT} x2={WIDTH - RIGHT} y1={y(price)} y2={y(price)} stroke="var(--market-border)" />
						<text x={LEFT - 10} y={y(price) + 4} textAnchor="end" fill="var(--market-muted)" fontSize="10">{price.toFixed(2)}</text>
					</g>;
				})}
				{candles.map((candle, index) => {
					const rising = candle.close >= candle.open;
					const color = rising ? "var(--market-bid)" : "var(--market-ask)";
					const flat = candle.open === candle.close;
					return <g key={candle.startStep}>
						<title>{`Steps ${candle.startStep}–${candle.endStep} · O ${candle.open.toFixed(2)} · H ${candle.high.toFixed(2)} · L ${candle.low.toFixed(2)} · C ${candle.close.toFixed(2)}${candle.samples < CANDLE_STEPS ? " · forming" : ""}`}</title>
						<line x1={x(index)} x2={x(index)} y1={y(candle.high)} y2={y(candle.low)} stroke={color} />
						{flat
							? <line x1={x(index) - bodyWidth / 2} x2={x(index) + bodyWidth / 2} y1={y(candle.open)} y2={y(candle.open)} stroke={color} />
							: <rect x={x(index) - bodyWidth / 2} y={y(Math.max(candle.open, candle.close))} width={bodyWidth} height={Math.abs(y(candle.open) - y(candle.close))} fill={rising ? "var(--market-surface)" : color} stroke={color} />}
						{index % labelEvery === 0 && <text x={x(index)} y={HEIGHT - BOTTOM + 18} textAnchor="middle" fill="var(--market-muted)" fontSize="10">{candle.startStep}</text>}
					</g>;
				})}
				{!last && <text x={WIDTH / 2} y={HEIGHT / 2} textAnchor="middle" fill="var(--market-muted)" fontSize="12">Waiting for the first simulation step</text>}
				<text x={LEFT + plotWidth / 2} y={HEIGHT - 6} textAnchor="middle" fill="var(--market-muted)" fontSize="11">Simulation step</text>
			</svg>
			<div className="market-chart-readout">
				{last ? `O ${last.open.toFixed(2)}   H ${last.high.toFixed(2)}   L ${last.low.toFixed(2)}   C ${last.close.toFixed(2)} · ${last.samples < CANDLE_STEPS ? "Forming" : "Closed"}` : "O —   H —   L —   C —"}
			</div>
		</div>
	);
}
