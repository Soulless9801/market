import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { buildMidPriceCandles, CHART_HISTORY_LIMIT } from "@/ui/candles";
import { CandlestickChart } from "@/ui/CandlestickChart";

describe("midprice candles", () => {
	it("uses real five-step OHLC and excludes the initial reference price", () => {
		expect(buildMidPriceCandles([500, 101, 105, 98, 99, 102, 104], 6)).toEqual([
			{ startStep: 1, endStep: 5, open: 101, high: 105, low: 98, close: 102, samples: 5 },
			{ startStep: 6, endStep: 6, open: 104, high: 104, low: 104, close: 104, samples: 1 },
		]);
	});

	it("updates only the forming candle until the next fixed bucket begins", () => {
		const prices = [100, 101, 102, 99, 98, 100, 104];
		const initial = buildMidPriceCandles(prices, 6);
		const updated = buildMidPriceCandles([...prices, 97], 7);
		expect(updated[0]).toEqual(initial[0]);
		expect(updated[1]).toEqual({ startStep: 6, endStep: 7, open: 104, high: 104, low: 97, close: 97, samples: 2 });
	});

	it("keeps 40 complete-or-forming candles stable as the price buffer scrolls", () => {
		const prices = Array.from({ length: 1_004 }, (_, step) => 100 + step % 7);
		for (let clock = 995; clock <= 1003; clock++) {
			const history = prices.slice(0, clock + 1);
			const full = buildMidPriceCandles(history, clock);
			const retained = buildMidPriceCandles(history.slice(-CHART_HISTORY_LIMIT), clock);
			expect(retained).toEqual(full);
			expect(retained).toHaveLength(40);
			expect(retained.slice(0, -1).every(candle => candle.samples === 5)).toBe(true);
		}
	});

	it("starts empty on reset and renders a useful waiting state", () => {
		expect(buildMidPriceCandles([100], 0)).toEqual([]);
		const html = renderToStaticMarkup(<CandlestickChart prices={[100]} currentStep={0} />);
		expect(html).toContain("Waiting for the first simulation step");
		expect(html).not.toMatch(/NaN|Infinity/);
	});

	it("renders flat candles without invalid geometry and distinguishes direction in grayscale", () => {
		const flat = renderToStaticMarkup(<CandlestickChart prices={[100, 100, 100]} currentStep={2} />);
		expect(flat).not.toMatch(/NaN|Infinity/);
		expect(flat).toContain("Forming");
		const both = renderToStaticMarkup(<CandlestickChart prices={[100, 100, 101, 102, 103, 104, 105, 104, 103, 102, 101]} currentStep={10} />);
		expect(both).toContain('stroke="var(--market-bid)"');
		expect(both).toContain('fill="var(--market-ask)"');
		expect(both).toContain("Closed");
	});
});
