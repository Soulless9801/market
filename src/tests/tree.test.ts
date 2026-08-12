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
		expect(tree.select(5)).toBeUndefined();
		expect(tree.has(100)).toBe(true);
		expect(tree.has(104)).toBe(false);
		expect(tree.toArray()).toEqual([99, 100, 101, 102, 103]);

		tree.insert(101);
		expect(tree.size).toBe(5);
		expect(tree.toArray()).toEqual([99, 100, 101, 102, 103]);

		tree.delete(101);
		expect(tree.size).toBe(4);
		expect(tree.toArray()).toEqual([99, 100, 102, 103]);
		expect(tree.has(101)).toBe(false);

		tree.delete(101);
		expect(tree.size).toBe(4);
		expect(tree.toArray()).toEqual([99, 100, 102, 103]);
		expect(tree.has(101)).toBe(false);

		tree.delete(99);
		tree.insert(99);
		expect(tree.size).toBe(4);
		expect(tree.toArray()).toEqual([99, 100, 102, 103]);
		expect(tree.has(99)).toBe(true);

		tree.delete(99);
		tree.delete(100);
		tree.delete(102);
		tree.delete(103);
		expect(tree.size).toBe(0);
		expect(tree.toArray()).toEqual([]);
		expect(tree.first()).toBeUndefined();
	});

	it("supports custom comparator for descending price retrieval", () => {
		const tree = new OrderedStatisticTree<number>((a, b) => b - a);
		const descendingValues = [99, 101, 100];
		descendingValues.forEach((value) => tree.insert(value));

		expect(tree.first()).toBe(101);
		expect(tree.toArray()).toEqual([101, 100, 99]);
	});

	it("supports custom comparator for complex objects", () => {
		interface Order {
			id: string;
			price: number;
		}

		const tree = new OrderedStatisticTree<Order>((a, b) => a.price - b.price);
		const orders: Order[] = [
			{ id: "order-1", price: 100 },
			{ id: "order-2", price: 99 },
			{ id: "order-3", price: 101 },
		];

		orders.forEach((order) => tree.insert(order));

		expect(tree.first()).toEqual({ id: "order-2", price: 99 });
		expect(tree.toArray()).toEqual([
			{ id: "order-2", price: 99 },
			{ id: "order-1", price: 100 },
			{ id: "order-3", price: 101 },
		]);
	});
});
