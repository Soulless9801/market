function relu(x: number): number {
    return Math.max(0, x);
}

function applyRelu(values: number[]): number[] {
    return values.map(relu);
}


export class DenseLayer {

    weights: number[][];
    biases: number[];

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

    train(input: number[], target: number[], learningRate: number): void {
        // Forward pass
        const hidden = applyRelu(
            this.hidden.forward(input)
        );
        const output = this.output.forward(hidden);

        // Compute loss (mean squared error)
        const loss = output.map((o, i) => o - target[i]);

        // Backpropagation (simplified for demonstration)
        const outputGradients = loss.map(l => l * 2);
        const hiddenGradients = this.output.weights.map((row, _) =>
            row.reduce((sum, w, j) => sum + w * outputGradients[j], 0)
        );

        // Update weights and biases (simplified)
        for (let i = 0; i < this.output.weights.length; i++) {
            for (let j = 0; j < this.output.weights[i].length; j++) {
                this.output.weights[i][j] -= learningRate * outputGradients[i] * hidden[j];
            }
            this.output.biases[i] -= learningRate * outputGradients[i];
        }

        for (let i = 0; i < this.hidden.weights.length; i++) {
            for (let j = 0; j < this.hidden.weights[i].length; j++) {
                this.hidden.weights[i][j] -= learningRate * hiddenGradients[i] * input[j];
            }
            this.hidden.biases[i] -= learningRate * hiddenGradients[i];
        }
    }
}
