import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PythonInferenceClient } from "@/runtime/python/PythonInferenceClient";
import { MAX_MESSAGE_BYTES } from "@/runtime/python/protocol";

vi.mock("node:child_process", () => ({ spawn: vi.fn() }));

class FakePython extends EventEmitter {
	stdin = new PassThrough();
	stdout = new PassThrough();
	stderr = new PassThrough();
	exitCode: number | null = null;
	signalCode: string | null = null;
	exitOnEnd = true;
	kill = vi.fn((signal: string) => {
		queueMicrotask(() => this.close(null, signal));
		return true;
	});
	constructor() {
		super();
		this.stdin.on("finish", () => {
			if (this.exitOnEnd) this.close(0);
		});
	}
	close(code: number | null, signal: string | null = null) {
		this.exitCode = code;
		this.signalCode = signal;
		this.emit("close", code, signal);
	}
	send(response: unknown) {
		this.stdout.write(JSON.stringify(response) + "\n");
	}
}

const clients: PythonInferenceClient[] = [];
function setup(
	options: Partial<
		ConstructorParameters<typeof PythonInferenceClient>[0]
	> = {},
) {
	const child = new FakePython();
	vi.mocked(spawn).mockReturnValue(
		child as unknown as ChildProcessWithoutNullStreams,
	);
	const client = new PythonInferenceClient({
		checkpoints: { release: "a path/checkpoint.pt" },
		...options,
	});
	clients.push(client);
	return { child, client };
}
async function ready(options = {}) {
	const pair = setup(options);
	const starting = pair.client.start();
	pair.child.send({ type: "ready", protocolVersion: 1 });
	await starting;
	return pair;
}
async function readRequest(child: FakePython) {
	await Promise.resolve();
	return JSON.parse(child.stdin.read().toString());
}

afterEach(async () => {
	await Promise.all(clients.splice(0).map((client) => client.shutdown()));
	vi.useRealTimers();
	vi.clearAllMocks();
});

describe("Python inference client", () => {
	it("waits for readiness, shares startup, uses safe arguments and spawns once", async () => {
		const { child, client } = setup({
			pythonExecutable: "/python with spaces",
			registryPath: "registry.json",
		});
		const starting = client.start();
		expect(client.start()).toBe(starting);
		let started = false;
		void starting.then(() => {
			started = true;
		});
		await Promise.resolve();
		expect(started).toBe(false);
		child.stdout.write('{"type":"rea');
		child.stdout.write('dy","protocolVersion":1}\n');
		await starting;
		await client.start();
		expect(spawn).toHaveBeenCalledTimes(1);
		expect(spawn).toHaveBeenCalledWith(
			"/python with spaces",
			[
				"-u",
				"-m",
				"ml.inference.server",
				"--ready-message",
				"--num-threads",
				"1",
				"--registry",
				"registry.json",
				"--model",
				"release=a path/checkpoint.pt",
			],
			expect.objectContaining({ shell: false }),
		);
	});

	it("serializes a snapshot and routes concurrent, out-of-order IDs across independent aliases", async () => {
		const { child, client } = await ready({
			checkpoints: {
				first: "a",
				second: "b",
				independent: "c",
			},
		});
		const input = [1, 2];
		const first = client.model("first").predict(input);
		input[0] = 99;
		const firstRequest = await readRequest(child);
		const second = client.model("second").predict([3]);
		const secondRequest = await readRequest(child);
		expect(firstRequest).toEqual({
			id: 0,
			type: "predict",
			model: "first",
			input: [1, 2],
		});
		child.stdout.write(
			JSON.stringify({
				id: secondRequest.id,
				type: "prediction",
				prediction: [4, 5],
			}) +
				"\n" +
				JSON.stringify({
					id: firstRequest.id,
					type: "prediction",
					prediction: [6, 7],
				}) +
				"\n",
		);
		await expect(first).resolves.toEqual([6, 7]);
		await expect(second).resolves.toEqual([4, 5]);
		const third = client.model("independent").predict([8]);
		const thirdRequest = await readRequest(child);
		child.send({
			id: thirdRequest.id,
			type: "prediction",
			prediction: [9, 10, 11, 12],
		});
		await expect(third).resolves.toEqual([9, 10, 11, 12]);
		expect(spawn).toHaveBeenCalledTimes(1);
	});

	it("rejects a correlated model error without ending subsequent predictions", async () => {
		const { child, client } = await ready();
		const failed = expect(
			client.predict("unknown", [1]),
		).rejects.toThrow("Unknown model");
		const request = await readRequest(child);
		child.send({
			id: request.id,
			type: "error",
			error: "Unknown model",
		});
		await failed;
		const next = client.predict("release", [2]);
		const nextRequest = await readRequest(child);
		child.send({
			id: nextRequest.id,
			type: "prediction",
			prediction: [3],
		});
		await expect(next).resolves.toEqual([3]);
	});

	it.each([
		"not-json\n",
		"null\n",
		"[]\n",
		'{"id":0,"type":"prediction","prediction":[]}\n',
		'{"id":0,"type":"prediction","prediction":[1e999]}\n',
		'{"id":0,"type":"prediction","prediction":["1"]}\n',
		'{"id":0,"type":"prediction","prediction":[1],"extra":true}\n',
		'{"id":true,"type":"prediction","prediction":[1]}\n',
		'{"id":999,"type":"prediction","prediction":[1]}\n',
		'{"id":null,"type":"error","error":"uncorrelated"}\n',
		'{"type":"ready","protocolVersion":1}\n',
	])(
		"rejects all pending calls on malformed/uncorrelated response: %s",
		async (line) => {
			const { child, client } = await ready();
			const first = expect(
				client.predict("release", [1]),
			).rejects.toThrow("protocol error");
			const second = expect(
				client.predict("release", [2]),
			).rejects.toThrow("protocol error");
			await Promise.resolve();
			child.stdout.write(line);
			await Promise.all([first, second]);
			await expect(
				client.predict("release", [3]),
			).rejects.toThrow("protocol error");
		},
	);

	it("rejects duplicate response IDs instead of satisfying a different request", async () => {
		const { child, client } = await ready();
		const first = client.predict("release", [1]);
		const request = await readRequest(child);
		child.send({
			id: request.id,
			type: "prediction",
			prediction: [2],
		});
		await first;
		const second = expect(
			client.predict("release", [3]),
		).rejects.toThrow("Unknown response ID");
		await readRequest(child);
		child.send({
			id: request.id,
			type: "prediction",
			prediction: [2],
		});
		await second;
	});

	it.each([Buffer.from([255, 10]), Buffer.alloc(MAX_MESSAGE_BYTES, 120)])(
		"rejects invalid UTF-8 and oversized frames",
		async (chunk) => {
			const { child, client } = await ready();
			const failed = expect(
				client.predict("release", [1]),
			).rejects.toThrow();
			await readRequest(child);
			child.stdout.write(chunk);
			await failed;
		},
	);

	it("exposes startup errors and stderr and rejects an incompatible handshake", async () => {
		const { child, client } = setup();
		const failed = expect(client.start()).rejects.toThrow(
			"checkpoint missing",
		);
		child.stderr.write("checkpoint missing");
		child.close(1);
		await failed;
		const incompatible = setup();
		const incompatibleStart = expect(
			incompatible.client.start(),
		).rejects.toThrow("protocol error");
		incompatible.child.send({ type: "ready", protocolVersion: 9 });
		await incompatibleStart;
	});

	it("rejects startup on spawn errors", async () => {
		const { child, client } = setup();
		const failed = expect(client.start()).rejects.toThrow("ENOENT");
		child.emit("error", new Error("ENOENT"));
		await failed;
	});

	it.each(["exit", "stdin", "stdout", "stderr"])(
		"rejects pending calls on %s failure",
		async (source) => {
			const { child, client } = await ready();
			const failed = expect(
				client.predict("release", [1]),
			).rejects.toThrow();
			await readRequest(child);
			if (source === "exit") {
				child.stdout.write('{"id":');
				child.close(2);
			} else
				child[
					source as "stdin" | "stdout" | "stderr"
				].emit("error", new Error("stream failed"));
			await failed;
		},
	);

	it("bounds startup and request waits and kills an unresponsive process", async () => {
		vi.useFakeTimers();
		const { child, client } = setup({ startupTimeoutMs: 100 });
		const failed = expect(client.start()).rejects.toThrow(
			"startup timed out",
		);
		await vi.advanceTimersByTimeAsync(100);
		await failed;
		expect(child.kill).toHaveBeenCalledWith("SIGTERM");
		const pair = await ready({ requestTimeoutMs: 100 });
		const timedOut = expect(
			pair.client.predict("release", [1]),
		).rejects.toThrow("timed out");
		await vi.advanceTimersByTimeAsync(100);
		await timedOut;
	});

	it("shuts down during startup and pending requests, and prevents reuse", async () => {
		const starting = setup();
		const cancelled = expect(
			starting.client.start(),
		).rejects.toThrow("shut down");
		await starting.client.shutdown();
		await cancelled;
		const { client } = await ready();
		const pending = expect(
			client.predict("release", [1]),
		).rejects.toThrow("shut down");
		await Promise.resolve();
		await Promise.all([client.shutdown(), client.shutdown()]);
		await pending;
		await expect(client.start()).rejects.toThrow("shut down");
	});

	it("forces shutdown after its deadline when stdin EOF is ignored", async () => {
		vi.useFakeTimers();
		const { child, client } = await ready({
			shutdownTimeoutMs: 100,
		});
		child.exitOnEnd = false;
		const stopping = client.shutdown();
		await vi.advanceTimersByTimeAsync(100);
		await stopping;
		expect(child.kill).toHaveBeenCalledWith("SIGKILL");
	});

	it("rejects invalid inputs before spawning and bounds pending requests", async () => {
		const { client } = setup();
		for (const input of [
			[],
			[NaN],
			[Infinity],
			[1e100],
			new Array(2),
		]) {
			await expect(
				client.predict("release", input),
			).rejects.toThrow();
		}
		expect(spawn).not.toHaveBeenCalled();
		const pair = await ready({ maxPendingRequests: 1 });
		const first = pair.client.predict("release", [1]);
		const request = await readRequest(pair.child);
		await expect(
			pair.client.predict("release", [2]),
		).rejects.toThrow("Too many pending");
		pair.child.send({
			id: request.id,
			type: "prediction",
			prediction: [3],
		});
		await first;
	});
});

it.each(["release\n", "release\r", "release\r\n"])(
	"rejects line endings in model aliases: %j",
	(alias) => {
		expect(() =>
			setup({ checkpoints: { [alias]: "model.pt" } }),
		).toThrow("Invalid model alias");
		expect(spawn).not.toHaveBeenCalled();
	},
);
