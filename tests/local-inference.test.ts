import { Readable } from "node:stream";
import type { IncomingMessage, ServerResponse } from "node:http";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createLocalInferenceHandler } from "@/runtime/python/localInferencePlugin";
import { bindLocalModel, loadLocalModel, loadLocalModelCatalog } from "@/runtime/browser/localModel";
import { FeatureManager, SIDE_ACTIONS } from "@/simulation";

function metadata() {
	const definition = FeatureManager.describe("mlp");
	return {
		inputShape: [definition.names.length], classNames: [...SIDE_ACTIONS], outputType: "logits",
		features: { ...definition, kind: "mlp", normalization: "none" },
	};
}

function backend() {
	return {
		getMetadata: vi.fn(async () => metadata()),
		getModels: vi.fn(async () => ({ "third-deployment": metadata(), "alpha-release": metadata(), "beta-release": metadata() })),
		predict: vi.fn(async () => [2, 1, 0]),
		shutdown: vi.fn(async () => {}),
	};
}

async function request(
	model = backend(), path = "predict", body = '{"input":[1,2]}',
	options: { origin?: string; address?: string; type?: string; host?: string } = {},
) {
	const incoming = Object.assign(Readable.from([Buffer.from(body)]), {
		url: `/__market/inference/${path}`, method: ["model", "models"].includes(path.split("?")[0]) ? "GET" : "POST",
		headers: { host: options.host ?? "localhost:5173", origin: options.origin, "content-type": options.type ?? "application/json" },
		socket: { remoteAddress: options.address ?? "127.0.0.1" },
	}) as unknown as IncomingMessage;
	const response = { writeHead: vi.fn(), end: vi.fn() };
	await createLocalInferenceHandler(model, "third-deployment")(incoming, response as unknown as ServerResponse, vi.fn());
	return { status: response.writeHead.mock.calls[0][0], body: JSON.parse(response.end.mock.calls[0][0]) };
}

afterEach(() => vi.unstubAllGlobals());

describe("local development inference bridge", () => {
	it("lists independent aliases and routes each prediction without changing other sessions", async () => {
		const model = backend();
		const catalog = await request(model, "models");
		expect(catalog.body.defaultAlias).toBe("third-deployment");
		expect(catalog.body.models.map((entry: { alias: string }) => entry.alias)).toEqual(["alpha-release", "beta-release", "third-deployment"]);
		expect((await request(model, "model?alias=beta-release")).body.alias).toBe("beta-release");
		for (const alias of ["alpha-release", "third-deployment", "beta-release", "alpha-release"]) {
			expect((await request(model, "predict", JSON.stringify({ model: alias, input: [1, 2] }))).status).toBe(200);
			expect(model.predict).toHaveBeenLastCalledWith(alias, [1, 2]);
		}
	});

	it.each(["unknown", "__proto__", null, {}, ""])("rejects invalid model selection %j", async alias => {
		const model = backend();
		expect((await request(model, "predict", JSON.stringify({ model: alias, input: [1] }))).status).toBe(400);
		expect(model.predict).not.toHaveBeenCalled();
	});

	it("rejects unknown metadata aliases and duplicate selections", async () => {
		expect((await request(backend(), "model?alias=missing")).status).toBe(400);
		expect((await request(backend(), "model?alias=alpha-release&alias=beta-release")).status).toBe(400);
	});
	it("serves checkpoint metadata and forwards raw features using the configured alias", async () => {
		const model = backend();
		expect((await request(model, "model")).body.alias).toBe("third-deployment");
		expect(model.getModels).toHaveBeenCalled();
		expect(await request(model)).toEqual({ status: 200, body: { prediction: [2, 1, 0] } });
		expect(model.predict).toHaveBeenCalledExactlyOnceWith("third-deployment", [1, 2]);
	});

	it("returns backend failures without producing a fallback prediction", async () => {
		const model = backend();
		model.predict.mockRejectedValue(new Error("Python exited"));
		expect(await request(model)).toEqual({ status: 503, body: { error: "Python exited" } });
	});

	it.each(['null', '{}', '{', '{"input":[]}', '{"input":["1"]}', '{"input":[1e999]}'])("rejects malformed input %s", async body => {
		const model = backend();
		expect((await request(model, "predict", body)).status).toBe(400);
		expect(model.predict).not.toHaveBeenCalled();
	});

	it("rejects oversized bodies, cross-origin callers, and nonlocal connections", async () => {
		const model = backend();
		expect((await request(model, "predict", "x".repeat(1_048_577))).status).toBe(413);
		expect((await request(model, "predict", undefined, { origin: "https://elsewhere.example" })).status).toBe(403);
		expect((await request(model, "predict", undefined, { address: "192.168.1.5" })).status).toBe(403);
		expect((await request(model, "predict", undefined, { host: "rebound.example:5173", origin: "http://rebound.example:5173" })).status).toBe(403);
		expect((await request(model, "predict", undefined, { type: "text/plain" })).status).toBe(415);
		expect(model.predict).not.toHaveBeenCalled();
	});
});

describe("browser model binding", () => {
	it("validates a catalog with a third independently named model", async () => {
		const catalog = { defaultAlias: "beta", models: ["alpha", "beta", "independent"].map(alias => ({ alias, metadata: metadata() })) };
		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(catalog)));
		expect(await loadLocalModelCatalog()).toEqual(JSON.parse(JSON.stringify(catalog)));
	});

	it.each([
		{ defaultAlias: "a", models: [] },
		{ defaultAlias: "missing", models: [{ alias: "a", metadata: metadata() }] },
		{ defaultAlias: "a", models: ["a", "a"].map(alias => ({ alias, metadata: metadata() })) },
	])("rejects ambiguous catalogs %j", async catalog => {
		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(catalog)));
		await expect(loadLocalModelCatalog()).rejects.toThrow();
	});

	it("keeps each binding pinned to its original alias", async () => {
		const fetchMock = vi.fn().mockImplementation(async () => Response.json({ prediction: [2, 1, 0] }));
		vi.stubGlobal("fetch", fetchMock);
		const descriptor = { alias: "alpha", metadata: metadata() };
		const first = bindLocalModel(descriptor);
		descriptor.alias = "beta";
		const second = bindLocalModel(descriptor);
		await first.model.predict([1]);
		await second.model.predict([2]);
		expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ input: [1], model: "alpha" });
		expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ input: [2], model: "beta" });
	});
	it("explains missing inference on a static-only host", async () => {
		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>app</html>", {
			headers: { "content-type": "text/html" },
		})));
		await expect(loadLocalModel()).rejects.toThrow("not configured on this server");
	});
	it("validates checkpoint metadata and predicts through the local endpoint", async () => {
		const fetchMock = vi.fn()
			.mockResolvedValueOnce(Response.json({ alias: "third-deployment", metadata: metadata() }))
			.mockResolvedValueOnce(Response.json({ prediction: [2, 1, 0] }));
		vi.stubGlobal("fetch", fetchMock);
		const binding = await loadLocalModel();
		expect(binding.alias).toBe("third-deployment");
		expect(binding.featureLayout).toBe("mlp");
		expect(await binding.model.predict([1, 2])).toEqual([2, 1, 0]);
		expect(fetchMock.mock.calls[1][1].body).toBe('{"input":[1,2],"model":"third-deployment"}');
	});

	it("rejects incompatible metadata before simulation starts", async () => {
		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({
			alias: "third-deployment", metadata: { ...metadata(), inputShape: [1] },
		})));
		await expect(loadLocalModel()).rejects.toThrow("incompatible");
	});

	it.each([{ prediction: [1, 2] }, { prediction: [1, null, 0] }, { prediction: [1, "2", 0] }])("rejects malformed prediction %j", async prediction => {
		vi.stubGlobal("fetch", vi.fn()
			.mockResolvedValueOnce(Response.json({ alias: "third-deployment", metadata: metadata() }))
			.mockResolvedValueOnce(Response.json(prediction)));
		const binding = await loadLocalModel();
		await expect(binding.model.predict([1])).rejects.toThrow("Invalid prediction");
	});

	it("surfaces Python failure to the simulation", async () => {
		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ error: "Python exited" }, { status: 503 })));
		await expect(loadLocalModel()).rejects.toThrow("Python exited");
	});
});
