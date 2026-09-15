interface TreeNode<T> {
	key: T;
	priority: number;
	size: number;
	left?: TreeNode<T>;
	right?: TreeNode<T>;
}

const getSize = <T>(node?: TreeNode<T>): number => node?.size ?? 0;

const recalculateSize = <T>(node: TreeNode<T>): void => {
	node.size = 1 + getSize(node.left) + getSize(node.right);
};

export class OrderedStatisticTree<T> {
	private root?: TreeNode<T>;
	private readonly compare: (a: T, b: T) => number;

	constructor(compare: (a: T, b: T) => number) {
		this.compare = compare;
	}

	get size(): number {
		return getSize(this.root);
	}

	has(key: T): boolean {
		let current = this.root;
		while (current) {
			const cmp = this.compare(key, current.key);
			if (cmp === 0) {
				return true;
			}
			current = cmp < 0 ? current.left : current.right;
		}
		return false;
	}

	insert(key: T): void {
		this.root = this.insertNode(this.root, key);
	}

	delete(key: T): void {
		this.root = this.deleteNode(this.root, key);
	}

	first(): T | undefined {
		return this.select(0);
	}

	select(rank: number): T | undefined {
		if (rank < 0 || rank >= this.size) {
			return undefined;
		}

		let current = this.root;
		let currentRank = rank;

		while (current) {
			const leftSize = getSize(current.left);
			if (currentRank < leftSize) {
				current = current.left;
				continue;
			}
			if (currentRank === leftSize) {
				return current.key;
			}

			currentRank -= leftSize + 1;
			current = current.right;
		}

		return undefined;
	}

	toArray(limit = Number.POSITIVE_INFINITY): T[] {
		const result: T[] = [];
		this.inOrder(this.root, result, limit);
		return result;
	}

	private inOrder(
		node: TreeNode<T> | undefined,
		out: T[],
		limit: number,
	): void {
		if (!node || out.length >= limit) {
			return;
		}

		this.inOrder(node.left, out, limit);
		if (out.length >= limit) {
			return;
		}

		out.push(node.key);
		this.inOrder(node.right, out, limit);
	}

	private insertNode(node: TreeNode<T> | undefined, key: T): TreeNode<T> {
		if (!node) {
			return {
				key,
				priority: Math.random(),
				size: 1,
			};
		}

		const cmp = this.compare(key, node.key);
		if (cmp === 0) {
			return node;
		}

		if (cmp < 0) {
			node.left = this.insertNode(node.left, key);
			if (node.left && node.left.priority < node.priority) {
				node = this.rotateRight(node);
			}
		} else {
			node.right = this.insertNode(node.right, key);
			if (node.right && node.right.priority < node.priority) {
				node = this.rotateLeft(node);
			}
		}

		recalculateSize(node);
		return node;
	}

	private deleteNode(
		node: TreeNode<T> | undefined,
		key: T,
	): TreeNode<T> | undefined {
		if (!node) {
			return undefined;
		}

		const cmp = this.compare(key, node.key);
		if (cmp < 0) {
			node.left = this.deleteNode(node.left, key);
			recalculateSize(node);
			return node;
		}

		if (cmp > 0) {
			node.right = this.deleteNode(node.right, key);
			recalculateSize(node);
			return node;
		}

		if (!node.left) {
			return node.right;
		}
		if (!node.right) {
			return node.left;
		}

		if (node.left.priority < node.right.priority) {
			node = this.rotateRight(node);
			node.right = this.deleteNode(node.right, key);
		} else {
			node = this.rotateLeft(node);
			node.left = this.deleteNode(node.left, key);
		}

		recalculateSize(node);
		return node;
	}

	private rotateLeft(node: TreeNode<T>): TreeNode<T> {
		const pivot = node.right;
		if (!pivot) {
			return node;
		}

		node.right = pivot.left;
		pivot.left = node;

		recalculateSize(node);
		recalculateSize(pivot);
		return pivot;
	}

	private rotateRight(node: TreeNode<T>): TreeNode<T> {
		const pivot = node.left;
		if (!pivot) {
			return node;
		}

		node.left = pivot.right;
		pivot.right = node;

		recalculateSize(node);
		recalculateSize(pivot);
		return pivot;
	}
}
