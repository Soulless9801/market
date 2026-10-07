import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { localInferencePlugin } from "./src/runtime/python/localInferencePlugin.ts";

export default defineConfig({
    plugins: [react(), localInferencePlugin({
		modelAlias: process.env.MARKET_MODEL_ALIAS,
		python: {
			pythonExecutable: process.env.MARKET_PYTHON ?? ".venv/bin/python",
			// A single-checkpoint override remains useful for training and smoke tests.
			...(process.env.MARKET_CHECKPOINT ? {
				checkpoints: { [process.env.MARKET_MODEL_ALIAS ?? "custom-model"]: process.env.MARKET_CHECKPOINT },
			} : {
				registryPath: process.env.MARKET_REGISTRY ?? "models/deployment/registry.json",
			}),
		},
	})],
	resolve: {
		tsconfigPaths: true
	},
	build: {
		rolldownOptions: {
			output: {
				codeSplitting: {
					minSize: 20_000,
					groups: [
						{
						name: "vendor",
						test: /node_modules/,
						},
					],
				},
			},
		},
	},
});
