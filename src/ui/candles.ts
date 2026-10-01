export const CANDLE_STEPS = 5;
export const MAX_VISIBLE_CANDLES = 40;
// Retain a full extra bucket so scrolling cannot change the oldest visible candle's open.
export const CHART_HISTORY_LIMIT = (MAX_VISIBLE_CANDLES + 1) * CANDLE_STEPS;

export interface PriceCandle {
	startStep: number;
	endStep: number;
	open: number;
	high: number;
	low: number;
	close: number;
	samples: number;
}

/** Aggregate sampled post-step midprices into fixed simulation-time buckets, not wall time. */
export function buildMidPriceCandles(prices: readonly number[], currentStep: number): PriceCandle[] {
	const candles: PriceCandle[] = [];
	const firstStep = currentStep - prices.length + 1;
	for (let index = 0; index < prices.length; index++) {
		const step = firstStep + index;
		// Step zero is the initial reference price, not a completed market observation.
		if (step <= 0) continue;
		const price = prices[index];
		const startStep = Math.floor((step - 1) / CANDLE_STEPS) * CANDLE_STEPS + 1;
		const previous = candles.at(-1);
		if (!previous || previous.startStep !== startStep) {
			candles.push({ startStep, endStep: step, open: price, high: price, low: price, close: price, samples: 1 });
		} else {
			previous.high = Math.max(previous.high, price);
			previous.low = Math.min(previous.low, price);
			previous.close = price;
			previous.endStep = step;
			previous.samples++;
		}
	}
	return candles.slice(-MAX_VISIBLE_CANDLES);
}
