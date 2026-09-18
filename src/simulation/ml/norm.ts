export const FEATURE_COUNT = 10;

export interface FeatureStatistics {
    means: number[];
    stds: number[];
}

export class FeatureNormalizer {
    readonly statistics: FeatureStatistics;

    constructor(statistics: FeatureStatistics) {
        if (
            statistics.means.length !== FEATURE_COUNT ||
            statistics.stds.length !== FEATURE_COUNT
        ) {
            throw new Error(`Expected ${FEATURE_COUNT} feature statistics.`);
        }

        this.statistics = {
            means: [...statistics.means],
            stds: [...statistics.stds],
        };
    }

    transform(features: number[]): number[] {
        if (features.length !== this.statistics.means.length) {
            throw new Error(`Expected ${this.statistics.means.length} features.`);
        }

        return features.map((value, index) =>
            (value - this.statistics.means[index]) /
                Math.max(this.statistics.stds[index], 1e-8),
        );
    }

    static fit(samples: number[][]): FeatureNormalizer {
        if (samples.length === 0 || samples.some((sample) => sample.length !== FEATURE_COUNT)) {
            throw new Error(`Expected non-empty samples with ${FEATURE_COUNT} features.`);
        }

        const means = Array.from({ length: FEATURE_COUNT }, (_, index) =>
            samples.reduce((sum, sample) => sum + sample[index], 0) / samples.length,
        );
        const stds = means.map((mean, index) =>
            Math.sqrt(
                samples.reduce((sum, sample) => sum + (sample[index] - mean) ** 2, 0) /
                samples.length,
            ),
        );

        return new FeatureNormalizer({ means, stds });
    }

    toJSON(): string {
        return JSON.stringify(this.statistics, null, 2);
    }

    static fromJSON(json: string): FeatureNormalizer {
        const statistics = JSON.parse(json) as FeatureStatistics;
        return new FeatureNormalizer(statistics);
    }
}