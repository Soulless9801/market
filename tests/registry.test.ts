import { describe, expect, it } from "vitest";
import { NamedRegistry } from "@/simulation/registry";

describe("registration identifiers", () => {
	it("rejects trailing line endings that JavaScript's regex end anchor otherwise accepts", () => {
		const registry = new NamedRegistry<number>("feature");
		for (const name of ["layout\n", "layout\r", "layout\r\n"]) {
			expect(() => registry.register(name, 1)).toThrow(
				"Invalid",
			);
		}
		expect(registry.names()).toEqual([]);
	});

	it("keeps independent registrations open while rejecting duplicate IDs", () => {
		const registry = new NamedRegistry<number>("feature");
		for (const [index, name] of [
			"first",
			"second",
			"independent-third",
		].entries())
			registry.register(name, index);
		expect(registry.get("independent-third")).toBe(2);
		expect(() => registry.register("first", 99)).toThrow(
			"already registered",
		);
		expect(() => registry.get("absent")).toThrow("Unknown");
	});
});
