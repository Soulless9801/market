import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
    plugins: [react()],
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
