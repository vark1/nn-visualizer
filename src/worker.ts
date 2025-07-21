import { trainModel, loss, Val } from "gradiatorjs";
import { createEngineModelFromVisualizer } from "./integration.js";

self.onmessage = async (event) => {
    const { type, rawVisData, x_train_data, x_train_shape, y_train_data, y_train_shape, trainingParams } = event.data;

    if (type !== 'start') return;

    console.log("worker: starting training");

    try {
        const x_train = new Val(x_train_shape);
        x_train.data = new Float64Array(x_train_data);

        const y_train = new Val(y_train_shape);
        y_train.data = new Float64Array(y_train_data);

        const [model, multiClass] = createEngineModelFromVisualizer(rawVisData, x_train);
        if (!model) {
            throw new Error("Model creation failed in the worker");
        };

        const params = {
            loss_fn: multiClass? loss.crossEntropyLoss_softmax: loss.crossEntropyLoss_binary,
            l_rate: trainingParams.l_rate,
            epochs: trainingParams.epochs,
            batch_size: trainingParams.batch_size,
            multiClass: multiClass
        };

        await trainModel(model, x_train, y_train, params, self);
    } catch (error: any){
        console.log("worker failed", error);
        self.postMessage({type: 'error', message: error.message});
    }
};