import { describe, expect, it } from "vitest";

import { OrderedStatisticTree } from "../engine";

describe("OrderedStatisticTree", () => {
	it("maintains sorted order, rank-select, and deletions", () => {
		const tree = new OrderedStatisticTree<number>((a, b) => a - b);
		const ascendingValues = [101, 99, 103, 100, 102];
		ascendingValues.forEach((value) => tree.insert(value));

		expect(tree.size).toBe(5);
		expect(tree.first()).toBe(99);
		expect(tree.select(2)).toBe(101);
		expect(tree.toArray()).toEqual([99, 100, 101, 102, 103]);

		tree.delete(101);
		expect(tree.size).toBe(4);
		expect(tree.toArray()).toEqual([99, 100, 102, 103]);
		expect(tree.has(101)).toBe(false);
	});

	it("supports custom comparator for descending best-price retrieval", () => {
		const tree = new OrderedStatisticTree<number>((a, b) => b - a);
		const descendingValues = [99, 101, 100];
		descendingValues.forEach((value) => tree.insert(value));

		expect(tree.first()).toBe(101);
		expect(tree.toArray()).toEqual([101, 100, 99]);
	});
});
