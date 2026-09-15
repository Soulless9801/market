import { describe, expect, it } from "vitest";

import { Deque } from "@/structs/Deque";

describe("Deque", () => {
	it("starts empty and returns undefined when reading or removing items", () => {
		const deque = new Deque<number>();

		expect(deque.isEmpty()).toBe(true);
		expect(deque.size()).toBe(0);
		expect(deque.peekFront()).toBeUndefined();
		expect(deque.peekBack()).toBeUndefined();
		expect(deque.popFront()).toBeUndefined();
		expect(deque.popBack()).toBeUndefined();
		expect(deque.size()).toBe(0);
	});

	it("supports FIFO operations from the front and back", () => {
		const deque = new Deque<string>();

		deque.pushBack("middle");
		deque.pushFront("front");
		deque.pushBack("back");

		expect(deque.size()).toBe(3);
		expect(deque.peekFront()).toBe("front");
		expect(deque.peekBack()).toBe("back");
		expect(deque.popFront()).toBe("front");
		expect(deque.popFront()).toBe("middle");
		expect(deque.popFront()).toBe("back");
		expect(deque.isEmpty()).toBe(true);
	});

	it("maintains both ends when removing from the back", () => {
		const deque = new Deque<number>();

		deque.pushFront(2);
		deque.pushFront(1);
		deque.pushBack(3);

		expect(deque.popBack()).toBe(3);
		expect(deque.peekFront()).toBe(1);
		expect(deque.peekBack()).toBe(2);
		expect(deque.popBack()).toBe(2);
		expect(deque.popBack()).toBe(1);
		expect(deque.size()).toBe(0);
		expect(deque.peekFront()).toBeUndefined();
		expect(deque.peekBack()).toBeUndefined();
	});

    it ("handles singular elements correctly", () => {
        const deque = new Deque<number>();
        deque.pushFront(42);

        expect(deque.size()).toBe(1);
        expect(deque.peekFront()).toBe(42);
        expect(deque.peekBack()).toBe(42);

        expect(deque.popBack()).toBe(42);
        expect(deque.isEmpty()).toBe(true);
    });
});