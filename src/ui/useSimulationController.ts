import { useCallback, useEffect, useRef, useState } from "react";
import { calculateMidPrice } from "@/engine";
import { buildAgents, Simulator } from "@/simulation";
import { loadLocalModel } from "@/runtime/browser/localModel";
import { buildMarketViewModel } from "./ViewModel";

const DEFAULT_SEED = 15;
const REFERENCE_PRICE = 100;

function createSimulator(seed: number, binding?: Awaited<ReturnType<typeof loadLocalModel>>) {
	return new Simulator({
		agents: buildAgents(seed, REFERENCE_PRICE, binding),
		referencePrice: REFERENCE_PRICE,
	});
}

function buildViewModel(simulator: Simulator, previousPrices: number[] = []) {
	const snapshot = simulator.getOrderBookSnapshot();
	return buildMarketViewModel(
		snapshot, simulator.getStatistics(), simulator.getTradeHistory(),
		simulator.getParticpantPortfolios(), simulator.getClock(),
		[...previousPrices.slice(-39), calculateMidPrice(snapshot, REFERENCE_PRICE)],
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
		setModelStatus("Loading local Python model…");
		try {
			if (!Number.isSafeInteger(nextSeed) || nextSeed < 0 || nextSeed > 0xffffffff)
				throw new Error("Seed must be an integer between 0 and 4294967295");
			const binding = import.meta.env.DEV ? await loadLocalModel() : undefined;
			if (generation !== generationRef.current) return;
			const simulator = createSimulator(nextSeed, binding);
			simulatorRef.current = simulator;
			setViewModel(buildViewModel(simulator));
			setModelStatus(binding
				? `Sandboxing · ${binding.alias}`
				: "Baseline");
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
				setViewModel(previous => buildViewModel(simulator, previous.midPriceSeries));
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
