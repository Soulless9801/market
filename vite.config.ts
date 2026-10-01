import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { localInferencePlugin } from "./src/runtime/python/localInferencePlugin.ts";

export default defineConfig({
    plugins: [react(), localInferencePlugin({
		modelAlias: process.env.MARKET_MODEL_ALIAS ?? "mlp-v1",
		python: {
			pythonExecutable: process.env.MARKET_PYTHON ?? ".venv/bin/python",
			checkpoints: {
				[process.env.MARKET_MODEL_ALIAS ?? "mlp-v1"]:
					process.env.MARKET_CHECKPOINT ?? "models/checkpoints/mlp-baseline.pt",
				["cnn-v2"]:
					process.env.MARKET_CHECKPOINT ?? "models/checkpoints/cnn-baseline-v2.pt",
			},
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
