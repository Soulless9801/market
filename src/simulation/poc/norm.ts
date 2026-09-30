import { NamedRegistry } from "@/simulation/registry";
// parent interface
export interface FeatureNormalizer {

    // fit normalizer to samples
    fit(samples: number[][]): void;

    // transform feature
    transform(features: number[]): number[];

    // convert normalizer to JSON
    toJSON(): string;

    // load normalizer from JSON
    fromJSON(json: string): void;
}

// gaussian normalizer statistics interface
interface GaussianStatistics {
    means: number[];
    stds: number[];
}

// gaussian normalizer class
export class GaussianNormalizer implements FeatureNormalizer {

    private statistics: GaussianStatistics = { means: [], stds: [] };
    private fc: number = 0;

    // @override
    fit(samples: number[][]): void {
        // do some checks
        if (samples.length === 0) throw new Error("No samples provided.");
        // fit to samples
        this.fc = samples[0].length;
        const means = Array.from({ length: this.fc }, (_, index) =>
            samples.reduce((sum, sample) => sum + sample[index], 0) / samples.length,
        );
        const stds = means.map((mean, index) =>
            Math.sqrt(
                samples.reduce((sum, sample) => sum + (sample[index] - mean) ** 2, 0) /
                samples.length,
            ),
        );
        this.statistics = { means, stds };
    }

    // @override
    transform(features: number[]): number[] {
        if (features.length !== this.fc) throw new Error(`Feature length mismatch. Expected ${this.fc}, got ${features.length}.`);

        return features.map((value, index) =>
            (value - this.statistics.means[index]) /
                Math.max(this.statistics.stds[index], 1e-8),
        );
    }

    // @override
    toJSON(): string {
        return JSON.stringify(this.statistics, null, 2);
    }

    // @override
    fromJSON(json: string): void {
        const statistics = JSON.parse(json) as GaussianStatistics;
        this.statistics = statistics;
        this.fc = statistics.means.length;
    }
}

// constructor type for normalizers
type NormalizerConstructor = new () => FeatureNormalizer;

// normalizer manager class
export class NormalizerManager {
	private static readonly registry =
		new NamedRegistry<NormalizerConstructor>("normalizer");
	static register(
		name: string,
		constructor: NormalizerConstructor,
	): void {
		this.registry.register(name, constructor);
	}
	static names(): string[] {
		return this.registry.names();
	}
	static getNormalizer(
		name: string,
		samples: number[][] | null,
	): FeatureNormalizer {
		const Constructor = this.registry.get(name);
		const normalizer = new Constructor();
		if (samples) normalizer.fit(samples);
		return normalizer;
	}
}
NormalizerManager.register("gaussian", GaussianNormalizer);
