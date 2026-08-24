import type { ObservableSimulatorContext } from '../simulator';

export function buildFeatures(context: ObservableSimulatorContext): number[] {
    return [
        (context.midPrice - context.recentMidPriceSeries[context.recentMidPriceSeries.length - 1]) / context.recentMidPriceSeries[context.recentMidPriceSeries.length - 1], // short return
        (context.midPrice - context.recentMidPriceSeries[0]) / context.recentMidPriceSeries[0], // long return
        context.spread / context.midPrice, // spread
        context.orderImbalance.imbalance, // imbalance
    ];
}

import { writeFile } from 'fs/promises';

export async function writeToFile(data: string, filePath: string): Promise<void> {
    try {
        await writeFile(filePath, data, 'utf-8');
        console.log('File written successfully.');
    } catch (error) {
        console.error('Error writing file:', error);
    }
}