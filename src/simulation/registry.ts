/** Small shared registry for explicit, local registrations. */
export class NamedRegistry<T> {
	private readonly entries = new Map<string, T>();
	private readonly category: string;
	constructor(category: string) {
		this.category = category;
	}
	register(name: string, value: T): void {
		if (!/^[a-z][a-z0-9_-]*$/.test(name))
			throw new Error(
				`Invalid ${this.category} name: ${name}`,
			);
		if (this.entries.has(name))
			throw new Error(
				`${this.category} already registered: ${name}`,
			);
		this.entries.set(name, value);
	}
	get(name: string): T {
		const value = this.entries.get(name);
		if (value === undefined)
			throw new Error(
				`Unknown ${this.category}: ${name}. Available: ${this.names().join(", ")}`,
			);
		return value;
	}
	names(): string[] {
		return [...this.entries.keys()];
	}
}
