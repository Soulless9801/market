import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin, PreviewServer, ViteDevServer } from "vite";
import { LOCAL_INFERENCE_PATH } from "../local-inference.ts";
import { PythonInferenceClient, type PythonInferenceOptions } from "./PythonInferenceClient.ts";

interface InferenceBackend {
	getMetadata(alias: string): Promise<unknown>;
	predict(alias: string, input: number[]): Promise<number[]>;
	shutdown(): Promise<void>;
}

export interface LocalInferenceOptions {
	modelAlias: string;
	python: PythonInferenceOptions;
}

function sendJson(response: ServerResponse, status: number, body: unknown) {
	response.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
	response.end(JSON.stringify(body));
}

/** Only the local Vite host owns Python. Checkpoint paths never come from browser requests. */
export function createLocalInferenceHandler(backend: InferenceBackend, modelAlias: string) {
	return async (request: IncomingMessage, response: ServerResponse, next: () => void) => {
		const path = request.url?.split("?")[0];
		if (path !== LOCAL_INFERENCE_PATH && !path?.startsWith(`${LOCAL_INFERENCE_PATH}/`)) {
			next();
			return;
		}
		// This bridge is for local development, not a publicly reachable inference service.
		const address = request.socket.remoteAddress;
		if (!["127.0.0.1", "::1", "::ffff:127.0.0.1"].includes(address ?? "")) {
			sendJson(response, 403, { error: "Local inference requires a loopback connection" });
			return;
		}
		// This middleware runs before Vite's own host checks; reject DNS rebinding here too.
		try {
			const hostname = new URL(`http://${request.headers.host}`).hostname;
			if (!["localhost", "127.0.0.1", "[::1]"].includes(hostname)) throw new Error();
		} catch {
			sendJson(response, 403, { error: "Local inference requires a localhost host" });
			return;
		}
		if (request.headers.origin) {
			try {
				if (new URL(request.headers.origin).host !== request.headers.host) throw new Error();
			} catch {
				sendJson(response, 403, { error: "Cross-origin inference is not allowed" });
				return;
			}
		}
		try {
			if (path === `${LOCAL_INFERENCE_PATH}/model` && request.method === "GET") {
				sendJson(response, 200, { alias: modelAlias, metadata: await backend.getMetadata(modelAlias) });
				return;
			}
			if (path !== `${LOCAL_INFERENCE_PATH}/predict` || request.method !== "POST") {
				sendJson(response, 404, { error: "Unknown inference endpoint" });
				return;
			}
			if (request.headers["content-type"] !== "application/json") {
				sendJson(response, 415, { error: "Expected application/json" });
				return;
			}
			const chunks: Buffer[] = [];
			let size = 0;
			for await (const chunk of request) {
				size += chunk.length;
				if (size > 1_048_576) {
					sendJson(response, 413, { error: "Inference request is too large" });
					return;
				}
				chunks.push(Buffer.from(chunk));
			}
			let body: unknown;
			try {
				body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
			} catch {
				sendJson(response, 400, { error: "Invalid JSON" });
				return;
			}
			const input = (body as { input?: unknown } | null)?.input;
			if (!Array.isArray(input) || !input.length || input.some(value => typeof value !== "number" || !Number.isFinite(value))) {
				sendJson(response, 400, { error: "Expected a nonempty finite feature vector" });
				return;
			}
			sendJson(response, 200, { prediction: await backend.predict(modelAlias, input) });
		} catch (error) {
			sendJson(response, 503, { error: error instanceof Error ? error.message : String(error) });
		}
	};
}

export function localInferencePlugin(options: LocalInferenceOptions): Plugin {
	let backend: PythonInferenceClient | undefined;
	const connect = (server: ViteDevServer | PreviewServer) => {
		// Dev and built-app preview use the same transport and checkpoint contract.
		// One lazily started Python process survives page reloads and simulation resets.
		backend = new PythonInferenceClient(options.python);
		const handler = createLocalInferenceHandler(backend, options.modelAlias);
		server.middlewares.use((request, response, next) => { void handler(request, response, next); });
		server.httpServer?.once("close", () => { void backend?.shutdown(); });
	};
	return {
		name: "market-local-inference",
		apply: "serve",
		configureServer: connect,
		configurePreviewServer: connect,
		async closeBundle() {
			await backend?.shutdown();
		},
	};
}
