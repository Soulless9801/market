import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { TextDecoder } from "node:util";
import type { PredictiveModel } from "@/simulation/ml/predictive-model";
import {
	MAX_MESSAGE_BYTES,
	parseResponse,
	validateModelAlias,
} from "./protocol";

export interface PythonInferenceOptions {
	pythonExecutable?: string;
	cwd?: string;
	checkpoints?: Readonly<Record<string, string>>;
	registryPath?: string;
	numThreads?: number;
	startupTimeoutMs?: number;
	requestTimeoutMs?: number;
	shutdownTimeoutMs?: number;
	maxPendingRequests?: number;
}

interface PendingPrediction {
	resolve: (prediction: number[]) => void;
	reject: (error: Error) => void;
	timer: ReturnType<typeof setTimeout>;
}

type ClientState =
	"idle" | "starting" | "ready" | "stopping" | "stopped" | "failed";

/** Node-only subprocess owner. One instance owns one process; create a new instance after shutdown. */
export class PythonInferenceClient {
	private readonly options: Required<
		Omit<PythonInferenceOptions, "registryPath">
	> & { registryPath?: string };
	private state: ClientState = "idle";
	private child?: ChildProcessWithoutNullStreams;
	private failure?: Error;
	private startPromise?: Promise<void>;
	private resolveStart?: () => void;
	private rejectStart?: (error: Error) => void;
	private startupTimer?: ReturnType<typeof setTimeout>;
	private killTimer?: ReturnType<typeof setTimeout>;
	private closed: Promise<void> = Promise.resolve();
	private resolveClosed?: () => void;
	private readonly pending = new Map<number, PendingPrediction>();
	// Never reuse IDs within a process: late or duplicated replies must not satisfy
	// a different caller's decision. A terminal failure requires a fresh client.
	private nextId = 0;
	private output = Buffer.alloc(0);
	private stderr = Buffer.alloc(0);
	private metadata: Record<string, unknown> = {};

	constructor(options: PythonInferenceOptions) {
		this.options = {
			pythonExecutable: options.pythonExecutable ?? "python",
			cwd: options.cwd ?? process.cwd(),
			checkpoints: { ...options.checkpoints },
			registryPath: options.registryPath,
			numThreads: options.numThreads ?? 1,
			startupTimeoutMs: options.startupTimeoutMs ?? 30_000,
			requestTimeoutMs: options.requestTimeoutMs ?? 30_000,
			shutdownTimeoutMs: options.shutdownTimeoutMs ?? 2_000,
			maxPendingRequests: options.maxPendingRequests ?? 128,
		};
		for (const key of [
			"numThreads",
			"startupTimeoutMs",
			"requestTimeoutMs",
			"shutdownTimeoutMs",
			"maxPendingRequests",
		] as const) {
			const value = this.options[key];
			if (
				!Number.isSafeInteger(value) ||
				value <= 0 ||
				value > 2_147_483_647
			) {
				throw new Error(
					`${key} must be a positive 32-bit integer`,
				);
			}
		}
		if (
			!this.options.registryPath &&
			Object.keys(this.options.checkpoints).length === 0
		) {
			throw new Error(
				"At least one checkpoint or registry path is required",
			);
		}
		for (const [alias, path] of Object.entries(
			this.options.checkpoints,
		)) {
			validateModelAlias(alias);
			if (typeof path !== "string" || !path.trim())
				throw new Error("Invalid checkpoint path");
		}
	}

	/** Resolves only after Python confirms that all checkpoints loaded successfully. */
	start(): Promise<void> {
		if (this.state === "ready") return Promise.resolve();
		if (this.state === "starting") return this.startPromise!;
		if (this.state !== "idle")
			return Promise.reject(
				this.failure ??
					new Error("Client is shut down"),
			);
		this.state = "starting";
		// Concurrent callers share readiness. The OS spawn event alone does not prove
		// Python finished importing modules or successfully loaded every checkpoint.
		this.startPromise = new Promise<void>((resolve, reject) => {
			this.resolveStart = resolve;
			this.rejectStart = reject;
		});
		this.closed = new Promise<void>((resolve) => {
			this.resolveClosed = resolve;
		});
		const args = [
			"-u",
			"-m",
			"ml.inference.server",
			"--ready-message",
			"--num-threads",
			String(this.options.numThreads),
		];
		if (this.options.registryPath)
			args.push("--registry", this.options.registryPath);
		for (const [alias, path] of Object.entries(
			this.options.checkpoints,
		))
			args.push("--model", `${alias}=${path}`);
		try {
			const child = spawn(
				this.options.pythonExecutable,
				args,
				{
					cwd: this.options.cwd,
					stdio: "pipe",
					shell: false,
				},
			);
			this.child = child;
			child.stdout.on("data", (chunk: Buffer) =>
				this.receive(chunk),
			);
			child.stderr.on("data", (chunk: Buffer) => {
				// Keep diagnostics bounded even if a backend logs continuously.
				this.stderr = Buffer.concat([
					this.stderr,
					chunk,
				]).subarray(-16_384);
			});
			child.on("error", (error) =>
				this.fail(
					new Error(
						`Python process error: ${error.message}`,
					),
				),
			);
			child.stdin.on("error", (error) =>
				this.fail(
					new Error(
						`Python input error: ${error.message}`,
					),
				),
			);
			child.stdout.on("error", (error) =>
				this.fail(
					new Error(
						`Python output error: ${error.message}`,
					),
				),
			);
			child.stderr.on("error", (error) =>
				this.fail(
					new Error(
						`Python diagnostic stream error: ${error.message}`,
					),
				),
			);
			child.on("close", (code, signal) => {
				if (
					this.state !== "stopping" &&
					this.state !== "failed"
				) {
					this.fail(
						new Error(
							`Python exited unexpectedly (code ${code}, signal ${signal})${this.output.length ? "; incomplete response" : ""}`,
						),
					);
				}
				clearTimeout(this.killTimer);
				clearTimeout(this.startupTimer);
				if (this.state === "stopping")
					this.state = "stopped";
				this.resolveClosed?.();
			});
			this.startupTimer = setTimeout(
				() =>
					this.fail(
						new Error(
							"Python startup timed out",
						),
					),
				this.options.startupTimeoutMs,
			);
		} catch (error) {
			this.fail(
				error instanceof Error
					? error
					: new Error(String(error)),
			);
			this.resolveClosed?.();
		}
		return this.startPromise;
	}

	/** Bind a deployment alias to the architecture-independent runtime interface. */
	model(alias: string): PredictiveModel {
		validateModelAlias(alias);
		return { predict: (input) => this.predict(alias, input) };
	}

	/** The consumer validates this checkpoint contract against its registered feature layout. */
	async getMetadata(alias: string): Promise<unknown> {
		validateModelAlias(alias);
		await this.start();
		if (!Object.hasOwn(this.metadata, alias))
			throw new Error(`No checkpoint metadata for ${alias}`);
		return structuredClone(this.metadata[alias]);
	}

	async predict(alias: string, input: number[]): Promise<number[]> {
		validateModelAlias(alias);
		// Copy before awaiting startup so callers cannot change an in-flight observation.
		if (
			!Array.isArray(input) ||
			input.length === 0 ||
			input.length > MAX_MESSAGE_BYTES
		)
			throw new Error("Invalid input vector");
		const snapshot = Array.from(input);
		if (
			!snapshot.every(
				(value) =>
					typeof value === "number" &&
					Number.isFinite(Math.fround(value)),
			)
		) {
			throw new Error(
				"Input must contain finite float32-compatible numbers",
			);
		}
		await this.start();
		if (this.state !== "ready")
			throw this.failure ?? new Error("Client is shut down");
		if (this.pending.size >= this.options.maxPendingRequests)
			throw new Error("Too many pending predictions");
		if (!Number.isSafeInteger(this.nextId))
			throw new Error(
				"Request IDs exhausted; create a new client",
			);
		const id = this.nextId++;
		const line =
			JSON.stringify({
				id,
				type: "predict",
				model: alias,
				input: snapshot,
			}) + "\n";
		if (Buffer.byteLength(line) > MAX_MESSAGE_BYTES)
			throw new Error("Request exceeds 1 MiB limit");
		return new Promise<number[]>((resolve, reject) => {
			const timer = setTimeout(
				() =>
					this.fail(
						new Error(
							`Prediction ${id} timed out`,
						),
					),
				this.options.requestTimeoutMs,
			);
			this.pending.set(id, { resolve, reject, timer });
			try {
				this.child!.stdin.write(line, (error) => {
					if (error)
						this.fail(
							new Error(
								`Python write failed: ${error.message}`,
							),
						);
				});
			} catch (error) {
				this.fail(
					new Error(
						`Python write failed: ${String(error)}`,
					),
				);
			}
		});
	}

	/** Reject outstanding calls, close stdin, and wait for exit (force-kill after the deadline). */
	shutdown(): Promise<void> {
		if (this.state === "idle") this.state = "stopped";
		if (
			this.state === "stopped" ||
			this.state === "stopping" ||
			this.state === "failed"
		)
			return this.closed;
		this.state = "stopping";
		this.rejectCalls(new Error("Client shut down"));
		this.child!.stdin.end();
		this.killTimer = setTimeout(
			() => this.child?.kill("SIGKILL"),
			this.options.shutdownTimeoutMs,
		);
		return this.closed;
	}

	private receive(chunk: Buffer): void {
		// Pipe chunks are not messages: one JSON line can span chunks, and one chunk
		// can contain several replies. Decode UTF-8 only after collecting a full line.
		if (this.state !== "starting" && this.state !== "ready") return;
		this.output = Buffer.concat([this.output, chunk]);
		let newline: number;
		while ((newline = this.output.indexOf(10)) >= 0) {
			if (newline + 1 > MAX_MESSAGE_BYTES)
				return this.fail(
					new Error(
						"Python response exceeds 1 MiB limit",
					),
				);
			const line = this.output.subarray(0, newline);
			this.output = this.output.subarray(newline + 1);
			try {
				const response = parseResponse(
					new TextDecoder("utf-8", {
						fatal: true,
					}).decode(line),
				);
				if (this.state === "starting") {
					if (response.type !== "ready")
						throw new Error(
							"Expected Python ready message",
						);
					this.state = "ready";
					this.metadata = structuredClone(
						response.models ?? {},
					);
					clearTimeout(this.startupTimer);
					this.resolveStart?.();
					continue;
				}
				if (
					response.type === "ready" ||
					response.id === null
				)
					throw new Error(
						"Uncorrelated Python response",
					);
				const pending = this.pending.get(response.id);
				if (!pending)
					throw new Error(
						`Unknown response ID: ${response.id}`,
					);
				this.pending.delete(response.id);
				clearTimeout(pending.timer);
				if (response.type === "prediction")
					pending.resolve(response.prediction);
				else pending.reject(new Error(response.error));
			} catch (error) {
				return this.fail(
					new Error(
						`Python protocol error: ${error instanceof Error ? error.message : String(error)}`,
					),
				);
			}
		}
		if (this.output.length >= MAX_MESSAGE_BYTES)
			this.fail(
				new Error(
					"Python response exceeds 1 MiB limit",
				),
			);
	}

	private rejectCalls(error: Error): void {
		clearTimeout(this.startupTimer);
		this.rejectStart?.(error);
		for (const pending of this.pending.values()) {
			clearTimeout(pending.timer);
			pending.reject(error);
		}
		this.pending.clear();
	}

	private fail(error: Error): void {
		// Do not retry decisions after transport failure: the server might already
		// have consumed a request, and replaying it could corrupt simulation ordering.
		if (
			this.state === "failed" ||
			this.state === "stopped" ||
			this.state === "stopping"
		)
			return;
		this.state = "failed";
		this.failure = new Error(
			`${error.message}${this.stderr.length ? `\nPython stderr:\n${this.stderr.toString("utf8")}` : ""}`,
			{ cause: error },
		);
		this.rejectCalls(this.failure);
		if (
			this.child &&
			this.child.exitCode === null &&
			this.child.signalCode === null
		) {
			this.child.kill("SIGTERM");
			this.killTimer = setTimeout(
				() => this.child?.kill("SIGKILL"),
				this.options.shutdownTimeoutMs,
			);
		}
	}
}
