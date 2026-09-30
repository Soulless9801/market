// Shared with the Python JSONL framing limit, including the terminating newline.
export const MAX_MESSAGE_BYTES = 1024 * 1024;

export type InferenceResponse =
	| {
			type: "ready";
			protocolVersion: 1;
			models?: Record<string, unknown>;
	  }
	| { type: "prediction"; id: number; prediction: number[] }
	| { type: "error"; id: number | null; error: string };

export function validateModelAlias(alias: string): void {
	if (
		typeof alias !== "string" ||
		!/^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$/.test(alias) ||
		/[\r\n]/.test(alias)
	) {
		throw new Error("Invalid model alias");
	}
}

export function parseResponse(line: string): InferenceResponse {
	// Treat subprocess output as an external boundary. Diagnostics belong on stderr;
	// a stray stdout message must fail clearly rather than resolve the wrong request.
	const response: unknown = JSON.parse(line);
	if (
		!response ||
		typeof response !== "object" ||
		Array.isArray(response)
	) {
		throw new Error("Response must be an object");
	}
	const fields = response as Record<string, unknown>;
	const keys = Object.keys(fields).sort().join(",");
	if (
		fields.type === "ready" &&
		(keys === "protocolVersion,type" ||
			keys === "models,protocolVersion,type") &&
		(fields.models === undefined ||
			(fields.models !== null &&
				typeof fields.models === "object" &&
				!Array.isArray(fields.models))) &&
		fields.protocolVersion === 1
	) {
		return {
			type: "ready",
			protocolVersion: 1,
			models: fields.models as
				Record<string, unknown> | undefined,
		};
	}
	const validId =
		typeof fields.id === "number" &&
		Number.isSafeInteger(fields.id) &&
		fields.id >= 0;
	if (
		fields.type === "prediction" &&
		keys === "id,prediction,type" &&
		validId &&
		Array.isArray(fields.prediction) &&
		fields.prediction.length > 0 &&
		fields.prediction.every(
			(value) =>
				typeof value === "number" &&
				Number.isFinite(value),
		)
	) {
		return fields as InferenceResponse;
	}
	if (
		fields.type === "error" &&
		keys === "error,id,type" &&
		(validId || fields.id === null) &&
		typeof fields.error === "string" &&
		fields.error.length > 0
	) {
		return fields as InferenceResponse;
	}
	throw new Error("Malformed inference response");
}
