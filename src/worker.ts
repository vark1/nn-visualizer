import { trainSingleBatch, loss, Val, state } from "gradiatorjs";
import { createEngineModelFromVisualizer } from "./integration.js";

self.onmessage = async (event) => {
    const { type, ...data } = event.data;

    switch(type) {
        case 'setup':
            console.log("Worker: Received 'setup' message.");
            const { rawVisData, x_train_data, x_train_shape, y_train_data, y_train_shape, trainingParams } = data;
            
            const x_train = new Val(x_train_shape);
            x_train.data = new Float64Array(x_train_data);
            const y_train = new Val(y_train_shape);
            y_train.data = new Float64Array(y_train_data);
            const [model, multiClass] = createEngineModelFromVisualizer(rawVisData, x_train);

            const params = {
                loss_fn: multiClass? loss.crossEntropyLoss_softmax: loss.crossEntropyLoss_binary,
                l_rate: trainingParams.l_rate,
                epochs: trainingParams.epochs,
                batch_size: trainingParams.batch_size,
                multiClass: multiClass
            };

            state.setupTrainingContext(model, x_train, y_train, params);
            
            self.postMessage({ type: 'setupComplete' });
            break;

        case 'tick':
            await trainSingleBatch(self);
            break;
        case 'stop':
            console.log("Worker: Received 'stop' message.");
            state.requestStopTraining();
            break;
        case 'pause':
            console.log("Worker: Received 'pause' message.");
            state.requestPause();
            break;
        case 'resume':
            console.log("Worker: Received 'resume' message.");
            state.requestResume();
            break;
    }
};