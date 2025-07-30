import { Val, trainModel, loss, state, model, layer } from "gradiatorjs";
import { TrainingProgress, visPackage, NetworkParams } from "./types.js";

import { prepareCatvnoncatData, prepareMNISTData } from "./utils/utils_datasets.js";
import { createEngineModelFromVisualizer } from "./integration.js";
import { renderNetworkGraph } from "./computational_graph.js";
import { NeuralNetworkVisualizer } from "./neuralNetworkVisualizer.js";
import { LossGraph } from "./loss_graph.js";

let currentModel: model.Sequential | null = null;
let VISUALIZER: NeuralNetworkVisualizer;
let lossGraph: LossGraph;

const startTrainingBtn = <HTMLButtonElement>document.getElementById('start-training-btn');
const pauseResumeBtn = <HTMLButtonElement>document.getElementById('pauseresume-btn');
const stopBtn = <HTMLButtonElement>document.getElementById('stop-btn');
const statusElement = <HTMLElement>document.getElementById('training-status');

const worker = new Worker(new URL('./worker.ts', import.meta.url), {
    type: 'module'
});

startTrainingBtn?.addEventListener('click', handleTraining);
pauseResumeBtn?.addEventListener('click', handlePauseResumeClick);
stopBtn?.addEventListener('click', handleStopClick);

document.addEventListener('DOMContentLoaded', () => {
    VISUALIZER = new NeuralNetworkVisualizer();
    lossGraph = new LossGraph('loss-accuracy-chart');
})

function updateBtnStates() {
    if (!startTrainingBtn || !pauseResumeBtn || !stopBtn) return;

    if (!state.getIsTraining()) {
        startTrainingBtn.style.display = 'block';
        startTrainingBtn.textContent = 'Start Training';
        stopBtn.style.display = 'none';
        pauseResumeBtn.style.display = 'none';
    } else {
        startTrainingBtn.style.display = 'none';
        stopBtn.style.display = 'block';
        pauseResumeBtn.style.display = 'block';

        if (state.getIsPaused()) {
            pauseResumeBtn.textContent = 'Resume';
        } else {
            pauseResumeBtn.textContent = 'Pause';
        }
    }
}

// document.getElementById('download-model-btn')?.addEventListener('click', () => {
//     if (!currentModel) {
//         alert("No trained model available to downlaod. please run training first");
//         return;
//     }
//     try {
//         const modelJSON = currentModel.toJSON();
//         downloadObjectAsJSON(modelJSON, 'trained_model.json');
//     } catch (error) {
//         console.error("Failed to serialize or download model:", error);
//         alert("Error: Could not download model. Check console for details.");
//     }
// });


function updateTrainingStatusUI(epoch: number, batch_idx: number, loss: number, accuracy: number, iterTime: number) {
    statusElement.textContent = `
    Epoch ${epoch + 1}, \n
    Batch ${batch_idx}: \n
    Loss=${loss.toFixed(4)}, 
    Acc=${accuracy.toFixed(2)}%, 
    Time per batch=${(iterTime/1000).toFixed(4)}s`

    if (lossGraph) {
        lossGraph.addData(batch_idx, loss, accuracy);
    }
}

function updateActivationVis(model: model.Sequential, visData: visPackage) {
    if (!VISUALIZER)                { console.error('Visualizer not found'); return; }

    const graphContainer = document.getElementById('graph-container')
    if (!graphContainer) return;

    const { sampleX, sampleY_label, layerOutputs } = visData;
    renderNetworkGraph(graphContainer, layerOutputs, model, sampleX, sampleY_label)
}

async function loadDataset() {
    const datasetType = (<HTMLSelectElement>document.getElementById('dataset-select')).value;
    let X = new Val([]);
    let Y = new Val([]);

    if (datasetType === 'mnist') {
        const [mnist_x_train, mnist_y_train] = await prepareMNISTData();
        X = mnist_x_train;
        Y = mnist_y_train;
    } else if (datasetType === 'catvnoncat') {
        const catvnoncat_data = await prepareCatvnoncatData();
        X = catvnoncat_data['train_x'];
        Y = catvnoncat_data['train_y'];
    }
    return [X, Y]
}

async function handleTraining() {
    if (!VISUALIZER) { console.error("Visulizer has not yet loaded for it to run the model."); return; }
    if (lossGraph) lossGraph.reset();

    if (state.getIsTraining()) {
        worker.postMessage({type: 'stop'});
        console.log("Training already in place.")
    }

    console.log("Main thread: Starting a new training session.")

    state.setTrainingState('TRAINING');
    updateBtnStates();

    const [x_train, y_train] = await loadDataset();

    const rawVisData = VISUALIZER.getNetworkConfig();                       // this model has nothing to do with the training itself, its being used for the visualizer
    [currentModel] = createEngineModelFromVisualizer(rawVisData, x_train);
    
    const params: NetworkParams = {
        l_rate: parseFloat((<HTMLInputElement>document.getElementById('learning-rate')).value) || 0.01,
        epochs: parseInt((<HTMLInputElement>document.getElementById('epoch')).value) || 500,
        batch_size: parseInt((<HTMLInputElement>document.getElementById('batch-size')).value) || 100,
    }

    console.log("starting training");
    worker.postMessage({
        type: 'setup',
        rawVisData: VISUALIZER.getNetworkConfig(),
        x_train_data: x_train.data.buffer,
        x_train_shape: x_train.shape,
        y_train_data: y_train.data.buffer,
        y_train_shape: y_train.shape,
        trainingParams: params
    },
    [x_train.data.buffer, y_train.data.buffer]);
}

function handleStopClick() {
    console.log("Main thread: Stop button clicked: Sending stop message to worker");
    worker.postMessage({type: 'stop'});
    statusElement.textContent = 'Training stopped';
    updateBtnStates();
    return;
}

function handlePauseResumeClick() {
    const isPaused = state.getIsPaused();
    console.log(`Main thread: Pause/Resume clicked. Currently paused: ${isPaused}`);

    if (isPaused) {
        worker.postMessage({ type: 'resume' });
    } else {
        worker.postMessage({ type: 'pause' });
    }
    updateBtnStates();
}

worker.onmessage = (event) => {
    const {type, epoch, batch_idx, loss, accuracy, iterTime, visData, reason} = event.data;

    switch (type) {
        case 'setupComplete':
            console.log("Main thread: Worker setup complete. Starting training ticker.");
            worker.postMessage({ type: 'tick' });
            state.setTrainingState('TRAINING');
            updateBtnStates();
            break;
        
        case 'batchEnd':
            const newSampleX = new Val(visData.sampleX.shape);
            newSampleX.data = new Float64Array(visData.sampleX.data);

            const reconstructedVisData = {
                sampleX: newSampleX,
                sampleY_label: visData.sampleY_label,
                layerOutputs: visData.layerOutputs.map((x: any) => {
                    const layerOutput: { Z?: Val, A?: Val } = {};

                    if (x.Zdata && x.Zshape) {
                        layerOutput.Z = new Val(x.Zshape);
                        layerOutput.Z.data = new Float64Array(x.Zdata);
                    }
                    if (x.Adata && x.Ashape) {
                        layerOutput.A = new Val(x.Ashape);
                        layerOutput.A.data = new Float64Array(x.Adata)
                    }
                    return layerOutput;
                })
            }

            updateTrainingStatusUI(epoch, batch_idx, loss, accuracy, iterTime);
            updateActivationVis(currentModel!, reconstructedVisData);

            if (state.getIsTraining() && !state.getStopTraining()) {
                worker.postMessage({ type: 'tick' });
            }
            break;

        case 'epochEnd':
            console.log("Main thread: 'epochEnd'.");
            if (state.getIsTraining()) {
                worker.postMessage({ type: 'tick' });
            }
            break;

        case 'paused':
            console.log("Main thread: Received 'paused' confirmation.");
            state.setTrainingState('PAUSED');
            updateBtnStates();
            break;

        case 'resumed':
            console.log("Main thread: Received 'resumed' confirmation.");
            state.setTrainingState('TRAINING');
            updateBtnStates();
            worker.postMessage({ type: 'tick' });
            break;

        case 'complete':
            console.log(`Training complete: ${reason}. Received 'complete' confirmation. Stopping ticker.`);
            state.setTrainingState('IDLE');
            updateBtnStates();
            break;
        
        case 'error':
            console.error("Received error(worker):", event.data.message);
            state.setTrainingState('IDLE');
            updateBtnStates();
            if (statusElement) statusElement.textContent = `Error: ${event.data.message || event.data}`;
            break;
    }
};