import { NamedRegistry } from "@/simulation/registry";
import type { AgentSide } from "@/simulation/agents";
import sideDatasetCnn from "@datasets/poc/side_dataset_cnn.json";
import sideDatasetMlp from "@datasets/poc/side_dataset_mlp.json";

export interface TrainingExample {
	features: number[];
	label: AgentSide;
}

export class DataLoader {
	// Local initialization is pure: unused legacy data can be removed from browser bundles.
	private static readonly registry = /* @__PURE__ */ (() => {
		const registry = new NamedRegistry<TrainingExample[]>(
			"legacy dataset",
		);
		registry.register("mlp", sideDatasetMlp as TrainingExample[]);
		registry.register("cnn", sideDatasetCnn as TrainingExample[]);
		return registry;
	})();
	static register(name: string, examples: TrainingExample[]): void {
		this.registry.register(name, examples);
	}
	static names(): string[] {
		return this.registry.names();
	}
	static getData(name: string): TrainingExample[] {
		return this.registry.get(name);
	}
}
