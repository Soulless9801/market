import { useCallback, useEffect, useRef, useState } from "react";
import { calculateMidPrice } from "@/engine";
import { buildAgents, Simulator } from "@/simulation";
import { loadLocalModel } from "@/runtime/browser/localModel";
import { BOOK_DEPTH, buildMarketViewModel } from "./ViewModel";
import { CHART_HISTORY_LIMIT } from "./candles";

const DEFAULT_SEED = 15;
const REFERENCE_PRICE = 100;

function createSimulator(seed: number, binding?: Awaited<ReturnType<typeof loadLocalModel>>) {
	return new Simulator({
		agents: buildAgents(seed, REFERENCE_PRICE, binding),
		referencePrice: REFERENCE_PRICE,
	});
}

function buildViewModel(simulator: Simulator) {
	const snapshot = simulator.getOrderBookSnapshot(BOOK_DEPTH);
	return buildMarketViewModel(
		snapshot, simulator.getStatistics(), simulator.getTradeHistory(),
		simulator.getParticpantPortfolios(), simulator.getClock(),
		[calculateMidPrice(snapshot, REFERENCE_PRICE)],
	);
}

export function useSimulationController() {
	const [initialSimulator] = useState(() => createSimulator(DEFAULT_SEED));
	const simulatorRef = useRef<Simulator | null>(null);
	const pendingRef = useRef(false);
	const generationRef = useRef(0);
	const [seed, setSeed] = useState(DEFAULT_SEED);
	const [isRunning, setIsRunning] = useState(true);
	const [playbackSpeed, setPlaybackSpeed] = useState(1);
	const [viewModel, setViewModel] = useState(() => buildViewModel(initialSimulator));
	const [modelStatus, setModelStatus] = useState("Loading…");
	const [error, setError] = useState<string | null>(null);
	const [isReady, setIsReady] = useState(false);

	const initialize = useCallback(async (nextSeed: number) => {
		const generation = ++generationRef.current;
		// Recreate agents to rewind their RNGs; discard results from an older reset.
		simulatorRef.current = null;
		setIsReady(false);
		setError(null);
		setModelStatus("Loading Python model…");
		try {
			if (!Number.isSafeInteger(nextSeed) || nextSeed < 0 || nextSeed > 0xffffffff)
				throw new Error("Seed must be an integer between 0 and 4294967295");
			const binding = await loadLocalModel();
			if (generation !== generationRef.current) return;
			const simulator = createSimulator(nextSeed, binding);
			simulatorRef.current = simulator;
			setViewModel(buildViewModel(simulator));
			setModelStatus(`Sandboxing · ${binding.alias}`);
			setIsReady(true);
		} catch (cause) {
			if (generation !== generationRef.current) return;
			setError(cause instanceof Error ? cause.message : String(cause));
		}
	}, []);

	useEffect(() => {
		// Promise scheduling also makes StrictMode cleanup invalidate the first load.
		let cancelled = false;
		const generation = generationRef;
		void Promise.resolve().then(() => { if (!cancelled) void initialize(DEFAULT_SEED); });
		return () => {
			cancelled = true;
			generation.current++;
			simulatorRef.current = null;
		};
	}, [initialize]);

	useEffect(() => {
		if (!isRunning || !isReady) return;
		const intervalId = window.setInterval(() => {
			const simulator = simulatorRef.current;
			if (!simulator || pendingRef.current) return;
			// A speed change or pause/resume must not launch a second pending step.
			pendingRef.current = true;
			void simulator.runStepAsync().then(() => {
				if (simulatorRef.current !== simulator) return;
				// Capture this completed step now; a queued React update must not sample a later step.
				const next = buildViewModel(simulator);
				setViewModel(previous => ({
					...next,
					midPriceSeries: [...previous.midPriceSeries.slice(-(CHART_HISTORY_LIMIT - 1)), next.midPriceSeries[0]],
				}));
			}).catch((cause: unknown) => {
				if (simulatorRef.current !== simulator) return;
				setError(cause instanceof Error ? cause.message : String(cause));
				setIsReady(false);
				setIsRunning(false);
			}).finally(() => { pendingRef.current = false; });
		}, 1000 / playbackSpeed);
		return () => window.clearInterval(intervalId);
	}, [isRunning, isReady, playbackSpeed]);

	return {
		viewModel, isRunning, seed, playbackSpeed, modelStatus, error, isReady,
		setSeed, setIsRunning, setPlaybackSpeed,
		reset: () => initialize(seed),
	};
}
