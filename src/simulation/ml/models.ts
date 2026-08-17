function relu(x: number): number {
    return Math.max(0, x);
}

function applyRelu(values: number[]): number[] {
    return values.map(relu);
}


export class DenseLayer {

    private readonly weights: number[][];
    private readonly biases: number[];

    constructor(weights: number[][], biases: number[]) {
        this.weights = weights;
        this.biases = biases;
    }

    forward(input: number[]): number[] {
        return this.weights.map((row, i) => {
            let value = this.biases[i];

            for (let j = 0; j < input.length; j++) {
                value += row[j] * input[j];
            }

            return value;
        });
    }
}

export class MLP {
    private readonly hidden: DenseLayer;
    private readonly output: DenseLayer;

    constructor() {
        this.hidden = new DenseLayer(
            [
                [0.1, -0.2, 0.3, 0.4],
                [-0.3, 0.2, 0.1, -0.4],
                [0.2, 0.1, -0.1, 0.3],
            ],
            [0, 0, 0],
        );

        this.output = new DenseLayer(
            [
                [0.2, -0.1, 0.3],
                [-0.2, 0.4, -0.1],
                [0.1, 0.2, 0.2],
            ],
            [0, 0, 0],
        );
    }

    predict(input: number[]): number[] {
        const hidden = applyRelu(
            this.hidden.forward(input)
        );

        return this.output.forward(hidden);
    }
}
