import type { AgentSide } from "@/simulation";
import sideDatasetCnn from "@datasets/side_dataset_cnn.json";
import sideDatasetMlp from "@datasets/side_dataset_mlp.json";

export interface TrainingExample {
	features: number[];
	label: AgentSide;
}

export class DataLoader {
	private static readonly registry = new Map<string, TrainingExample[]>([
		["mlp", sideDatasetMlp as TrainingExample[]],
		["cnn", sideDatasetCnn as TrainingExample[]],
	]);

	static getData(modelName: string): TrainingExample[] {
		const data = this.registry.get(modelName);
		if (!data) {
			throw new Error(
				`No dataset registered for model: ${modelName}`,
			);
		}
		return data;
	}
}
