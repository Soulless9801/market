/** Runtime inference is asynchronous and independent of training or architecture. */
export interface PredictiveModel {
	/** Accept raw features; return class scores. The implementation owns preprocessing. */
	predict(input: number[]): Promise<number[]>;
}
