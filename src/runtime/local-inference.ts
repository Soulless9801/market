// Browser and development server share routes without importing Node into the UI.
export const LOCAL_INFERENCE_PATH = "/__market/inference";

export interface LocalModelDescriptor {
	alias: string;
	metadata: unknown;
}
