interface DequeNode<T> {
    value: T;
    next: DequeNode<T> | undefined;
    prev: DequeNode<T> | undefined;
}

// deque class w/ linkedlist implementation
export class Deque<T> {

    private head: DequeNode<T> | undefined;
    private tail: DequeNode<T> | undefined;

    private count : number = 0;

    pushFront(item: T): void {
        const node: DequeNode<T> = {
            value: item,
            next: this.head,
            prev: undefined,
        };

        if (this.head) {
            this.head.prev = node;
        } else {
            this.tail = node;
        }

        this.head = node;
        this.count++;
    }

    pushBack(item: T): void {
        const node: DequeNode<T> = {
            value: item,
            next: undefined,
            prev: this.tail,
        };

        if (this.tail) {
            this.tail.next = node;
        } else {
            this.head = node;
        }

        this.tail = node;
        this.count++;
    }

    popFront(): T | undefined {
        if (!this.head) {
            return undefined;
        }

        const value = this.head.value;
        this.head = this.head.next;

        if (this.head) {
            this.head.prev = undefined;
        } else {
            this.tail = undefined;
        }

        this.count--;
        return value;
    }

    popBack(): T | undefined {
        if (!this.tail) {
            return undefined;
        }

        const value = this.tail.value;
        this.tail = this.tail.prev;

        if (this.tail) {
            this.tail.next = undefined;
        } else {
            this.head = undefined;
        }

        this.count--;
        return value;
    }

    peekFront(): T | undefined {
        return this.head?.value;
    }

    peekBack(): T | undefined {
        return this.tail?.value;
    }

    isEmpty(): boolean {
        return this.count === 0;
    }

    size(): number {
        return this.count;
    }

    toArray(): T[] {
        const result: T[] = [];
        let current = this.head;

        while (current) {
            result.push(current.value);
            current = current.next;
        }

        return result;
    }

    clear(): void {
        this.head = undefined;
        this.tail = undefined;
        this.count = 0;
    }
}