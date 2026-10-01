export function formatPrice(value: number | null): string {
	return value === null ? "—" : "$" + value.toFixed(2);
}

export function formatQuantity(value: number): string {
	return value.toLocaleString();
}
